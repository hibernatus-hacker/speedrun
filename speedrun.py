#!/usr/bin/env python3
"""
speedrun - Simple tool to run ML training on vast.ai GPU instances
"""

import os
import sys
import json
import time
import tarfile
import tempfile
import subprocess
from pathlib import Path
from typing import Dict

import paramiko
from scp import SCPClient


class VastAI:
    """Simple vast.ai CLI wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def _run_vastai_cmd(self, cmd: list) -> dict:
        """Run a vastai CLI command and return parsed JSON result"""
        try:
            result = subprocess.run(
                ["vastai"] + cmd + ["--raw"],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"vastai command failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse vastai response: {e}")
    
    def find_best_gpu(self) -> Dict:
        """Find the most powerful multi-GPU instance"""
        print("🔍 Finding most powerful multi-GPU instance on vast.ai...")
        
        # First try to find instances with 4+ GPUs under $10/hr
        instances = self._run_vastai_cmd([
            "search", "offers", 
            "rentable=true reliability>0.95 num_gpus>=4 dph_total<10.0",
            "--order", "num_gpus-,gpu_total_ram-",
            "--limit", "10"
        ])
        
        # If no affordable 4+ GPU instances, try any 4+ GPU instances
        if not instances:
            print("🔍 No affordable 4+ GPU instances found, searching for any 4+ GPU instances...")
            instances = self._run_vastai_cmd([
                "search", "offers", 
                "rentable=true reliability>0.95 num_gpus>=4",
                "--order", "num_gpus-,gpu_total_ram-",
                "--limit", "10"
            ])
        
        # If no 4+ GPU instances, fall back to any multi-GPU (2+)
        if not instances:
            print("🔍 No 4+ GPU instances found, searching for 2+ GPU instances...")
            instances = self._run_vastai_cmd([
                "search", "offers", 
                "rentable=true reliability>0.95 num_gpus>=2",
                "--order", "num_gpus-,gpu_total_ram-",
                "--limit", "10"
            ])
        
        # If still no multi-GPU instances, fall back to single GPU
        if not instances:
            print("🔍 No multi-GPU instances found, searching for single GPU instances...")
            instances = self._run_vastai_cmd([
                "search", "offers", 
                "rentable=true reliability>0.95",
                "--order", "gpu_total_ram-",
                "--limit", "10"
            ])
        
        if not instances:
            raise RuntimeError("No GPU instances found")
        
        # Take the first (most GPUs + highest total GPU RAM) instance
        best = instances[0]
        
        gpu_ram_gb = best['gpu_ram'] / 1024  # Convert MB to GB
        total_gpu_ram_gb = best['gpu_total_ram'] / 1024  # Convert MB to GB
        num_gpus = best['num_gpus']
        
        print(f"✅ Selected: {num_gpus}x {best['gpu_name']} ({gpu_ram_gb:.1f}GB each, {total_gpu_ram_gb:.1f}GB total) - ${best['dph_total']:.2f}/hr")
        return best
    
    def create_instance(self, instance_id: int) -> int:
        """Create a new instance"""
        print("🚀 Creating instance...")
        
        result = self._run_vastai_cmd([
            "create", "instance", str(instance_id),
            "--image", "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
            "--disk", "50",
            "--label", "speedrun",
            "--ssh",
            "--direct"
        ])
        
        contract_id = result["new_contract"]
        print(f"✅ Instance created: {contract_id}")
        return contract_id
    
    def wait_for_instance(self, contract_id: int) -> Dict:
        """Wait for instance to be ready"""
        print("⏳ Waiting for instance to be ready...")
        
        while True:
            instances = self._run_vastai_cmd(["show", "instances"])
            
            for instance in instances:
                if instance["id"] == contract_id and instance["actual_status"] == "running":
                    print("✅ Instance is ready!")
                    # Debug: show available fields for SSH connection
                    ssh_fields = [k for k in instance.keys() if any(term in k.lower() for term in ['ssh', 'port', 'host', 'ip'])]
                    if ssh_fields:
                        print(f"🔍 Available SSH-related fields: {ssh_fields}")
                    return instance
            
            time.sleep(10)
    
    def destroy_instance(self, contract_id: int):
        """Destroy an instance"""
        print(f"🗑️ Destroying instance {contract_id}...")
        self._run_vastai_cmd(["destroy", "instance", str(contract_id)])
        print("✅ Instance destroyed")


class SpeedRun:
    """Main speedrun orchestrator"""
    
    def __init__(self, api_key: str):
        self.vast = VastAI(api_key)
    
    def package_project(self, project_path: Path) -> Path:
        """Package project into tar.gz"""
        print("📦 Packaging project...")
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        with tarfile.open(temp_path, 'w:gz') as tar:
            for item in project_path.iterdir():
                if not item.name.startswith('.'):
                    tar.add(item, arcname=item.name)
        
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        print(f"✅ Packaged project ({size_mb:.1f}MB)")
        return temp_path
    
    def run_on_instance(self, instance_info: Dict, package_path: Path, project_name: str):
        """Upload, run training, and download results"""
        host = instance_info["public_ipaddr"]
        
        # Try to find SSH port in different possible locations
        port = 22  # Default SSH port
        
        # Method 1: Check if direct connection info is available
        if "ssh_host" in instance_info and "ssh_port" in instance_info:
            host = instance_info["ssh_host"] 
            port = instance_info["ssh_port"]
        # Method 2: Check traditional ports structure
        elif "ports" in instance_info and "22/tcp" in instance_info["ports"]:
            port_info = instance_info["ports"]["22/tcp"]
            if isinstance(port_info, list) and len(port_info) > 0:
                port = port_info[0]["HostPort"]
            else:
                port = port_info
        # Method 3: Check for direct_port_start (common in vast.ai)
        elif "direct_port_start" in instance_info:
            port = instance_info["direct_port_start"]
        # Method 4: Use default port 22
        else:
            print("⚠️ No specific SSH port found, trying default port 22")
        
        # Connect via SSH
        print(f"🔗 Connecting to {host}:{port}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username="root", password="root", timeout=60)
        scp = SCPClient(ssh.get_transport())
        print("✅ Connected via SSH")
        
        try:
            # Upload project
            print("📤 Uploading project...")
            scp.put(str(package_path), "/root/project.tar.gz")
            
            # Extract and setup
            print("📂 Extracting project...")
            stdin, stdout, stderr = ssh.exec_command("cd /root && tar -xzf project.tar.gz && rm project.tar.gz")
            stdout.read()  # Wait for command to complete
            
            # Check what directory was created
            stdin, stdout, stderr = ssh.exec_command("ls -la /root/")
            dir_list = stdout.read().decode()
            print(f"🔍 Directory contents: {dir_list}")
            
            # Find the actual project directory (it might be named differently)
            stdin, stdout, stderr = ssh.exec_command("find /root -name 'train.py' -type f")
            train_py_path = stdout.read().decode().strip()
            if train_py_path:
                actual_project_dir = '/'.join(train_py_path.split('/')[:-1])
                print(f"🔍 Found train.py at: {actual_project_dir}")
            else:
                actual_project_dir = f"/root/{project_name}"
                print(f"⚠️ train.py not found, using default: {actual_project_dir}")
            
            # Install requirements if they exist
            stdin, stdout, stderr = ssh.exec_command(f"ls {actual_project_dir}/requirements.txt")
            if not stderr.read():
                print("📦 Installing requirements...")
                stdin, stdout, stderr = ssh.exec_command(f"cd {actual_project_dir} && pip install -r requirements.txt")
                stdout.read()  # Wait for command to complete
                print("✅ Requirements installed")
            
            # Run training
            print("🚀 Running train.py...")
            stdin, stdout, stderr = ssh.exec_command(f"cd {actual_project_dir} && python train.py", get_pty=True)
            
            # Stream output
            for line in iter(stdout.readline, ''):
                print(f"  {line.rstrip()}")
            
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError("Training failed")
            
            print("✅ Training completed")
            
            # Download model artifacts
            print("📥 Downloading model artifacts...")
            
            # Find all model files
            patterns = ["*.pt", "*.pth", "*.pkl", "*.h5", "*.hdf5"]
            model_files = []
            
            for pattern in patterns:
                stdin, stdout, stderr = ssh.exec_command(f"find {actual_project_dir} -name '{pattern}' -type f")
                files = [f.strip() for f in stdout.read().decode().split('\n') if f.strip()]
                model_files.extend(files)
            
            if model_files:
                # Create results directory
                results_dir = Path(f"speedrun_results_{int(time.time())}")
                results_dir.mkdir(exist_ok=True)
                
                # Download each file
                for remote_file in model_files:
                    filename = Path(remote_file).name
                    local_file = results_dir / filename
                    print(f"  Downloading {filename}...")
                    scp.get(remote_file, str(local_file))
                
                print(f"✅ Downloaded {len(model_files)} files to {results_dir}")
            else:
                print("⚠️ No model artifacts found")
        
        finally:
            ssh.close()
    
    def run(self, project_path: Path):
        """Main execution flow"""
        start_time = time.time()
        contract_id = None
        
        try:
            # Find and create instance
            instance = self.vast.find_best_gpu()
            contract_id = self.vast.create_instance(instance["id"])
            instance_info = self.vast.wait_for_instance(contract_id)
            
            # Package project
            package_path = self.package_project(project_path)
            
            try:
                # Run training
                self.run_on_instance(instance_info, package_path, project_path.name)
                
                # Calculate cost
                duration_hours = (time.time() - start_time) / 3600
                cost = duration_hours * instance["dph_total"]
                print(f"\n💰 Total cost: ${cost:.2f}")
                
            finally:
                package_path.unlink()  # Clean up
        
        finally:
            if contract_id:
                self.vast.destroy_instance(contract_id)


def main():
    """Main entry point"""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python speedrun.py /path/to/your/project [--dry-run]")
        sys.exit(1)
    
    project_path = Path(sys.argv[1]).resolve()
    dry_run = len(sys.argv) == 3 and sys.argv[2] == "--dry-run"
    
    if not project_path.exists() or not project_path.is_dir():
        print(f"Error: '{project_path}' is not a valid directory")
        sys.exit(1)
    
    if not (project_path / "train.py").exists():
        print("Error: No train.py found in project directory")
        sys.exit(1)
    
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("Error: VAST_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"\n🚀 Starting speedrun for: {project_path.name}")
    if dry_run:
        print("🔍 DRY RUN MODE - will only search for GPUs, not create instances")
    print("=" * 50)
    
    runner = SpeedRun(api_key)
    
    if dry_run:
        # Just test the search functionality
        try:
            runner.vast.find_best_gpu()
            print("\n✅ Dry run completed - speedrun is working!")
        except Exception as e:
            print(f"\n❌ Dry run failed: {e}")
            sys.exit(1)
    else:
        runner.run(project_path)
        print("\n✨ Speedrun completed!")


if __name__ == "__main__":
    main()