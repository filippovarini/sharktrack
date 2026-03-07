#!/usr/bin/env python3
"""
Cross-Platform Performance Optimizer for BRUV Video Processing
Automatically optimizes system settings for maximum CPU performance
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class PerformanceOptimizer:
    """Cross-platform system performance optimization"""

    def __init__(self):
        self.system = platform.system().lower()
        self.is_admin = self._check_admin_privileges()
        self.optimization_commands = []
        self.restore_commands = []

    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator/root privileges"""
        if self.system == "windows":
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            except:
                return False
        else:
            return os.geteuid() == 0

    def check_cpu_governor(self) -> Dict[str, str]:
        """Check current CPU governor settings (Linux only)"""
        if self.system != "linux":
            return {"status": "not_applicable", "message": "CPU governor check only available on Linux"}

        try:
            print("   🔍 Checking CPU governor settings...", end="", flush=True)

            # Check if scaling governor exists
            governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if not governor_path.exists():
                print(" not available")
                return {"status": "unavailable", "message": "CPU frequency scaling not available"}

            # Get current governor from first CPU (they should all be the same)
            with open(governor_path, 'r') as f:
                governor = f.read().strip()

            # Quick check if all CPUs have the same governor
            cpu_paths = list(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
            all_same = True
            for cpu_path in cpu_paths[:5]:  # Check first 5 CPUs
                try:
                    with open(cpu_path, 'r') as f:
                        if f.read().strip() != governor:
                            all_same = False
                            break
                except:
                    continue

            print(f" {governor}")

            if all_same:
                return {
                    "status": "detected",
                    "current": governor,
                    "message": f"All CPUs using {governor} governor",
                    "count": len(cpu_paths)
                }
            else:
                return {
                    "status": "mixed",
                    "message": f"Mixed governors detected (primary: {governor})",
                    "count": len(cpu_paths)
                }

        except subprocess.TimeoutExpired:
            print(" timeout")
            return {"status": "timeout", "message": "CPU governor check timed out"}
        except Exception as e:
            print(f" error")
            return {"status": "error", "message": f"Error checking CPU governor: {e}"}

    def check_thermal_status(self) -> Dict[str, str]:
        """Check thermal throttling status"""
        if self.system == "linux":
            try:
                print("   🌡️  Checking thermal sensors...", end="", flush=True)

                # Try different thermal monitoring approaches
                thermal_info = {}

                # Method 1: Check thermal zones (with timeout)
                thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
                if thermal_zones:
                    for zone in thermal_zones[:3]:  # Check first 3 zones
                        try:
                            with open(zone / "temp") as f:
                                temp = int(f.read().strip()) / 1000  # Convert from millicelsius
                            with open(zone / "type") as f:
                                zone_type = f.read().strip()
                            thermal_info[zone_type] = f"{temp:.1f}°C"
                        except:
                            continue

                # Method 2: Try sensors command (with timeout)
                try:
                    result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        thermal_info["sensors_output"] = "Available"
                except:
                    pass

                print(f" found {len(thermal_info)} sensors")

                return {
                    "status": "detected" if thermal_info else "limited",
                    "thermal_zones": thermal_info,
                    "message": f"Found {len(thermal_info)} thermal sensors"
                }

            except Exception as e:
                print(" error")
                return {"status": "error", "message": f"Error checking thermal status: {e}"}

        elif self.system == "darwin":  # macOS
            try:
                result = subprocess.run(['sysctl', 'machdep.xcpm.cpu_thermal_state'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return {"status": "detected", "message": "macOS thermal monitoring available"}
            except:
                pass

        return {"status": "not_available", "message": "Thermal monitoring not available on this platform"}

    def _check_gpu_info(self) -> Optional[Dict[str, str]]:
        """Check GPU hardware information"""
        gpu_info = {}

        try:
            # Try nvidia-smi first for NVIDIA GPUs
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    gpu_data = lines[0].split(',')
                    gpu_name = gpu_data[0].strip()
                    gpu_info['model'] = gpu_name

                    # Try to get core count for NVIDIA
                    try:
                        core_result = subprocess.run(['nvidia-smi', '--query-gpu=count',
                                                    '--format=csv,noheader,nounits'],
                                                   capture_output=True, text=True, timeout=3)
                        if core_result.returncode == 0:
                            # Note: nvidia-smi doesn't directly report CUDA cores, so we skip this
                            pass
                    except:
                        pass

                    return gpu_info
        except:
            pass

        # Fallback to lspci for basic GPU detection
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line and ('NVIDIA' in line or 'AMD' in line):
                        # Extract GPU name
                        if 'NVIDIA' in line:
                            gpu_name = line.split('NVIDIA Corporation')[-1].strip()
                            gpu_info['model'] = f"NVIDIA {gpu_name.split('(')[0].strip()}"
                        elif 'AMD' in line:
                            gpu_name = line.split('Advanced Micro Devices')[-1].strip()
                            gpu_info['model'] = f"AMD {gpu_name.split('(')[0].strip()}"

                        return gpu_info
        except:
            pass

        return None

    def _check_gpu_governor(self) -> Optional[str]:
        """Check GPU governor/performance state"""
        try:
            # NVIDIA GPU performance state
            result = subprocess.run(['nvidia-smi', '--query-gpu=performance.state',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                perf_state = result.stdout.strip()
                return f"Performance state {perf_state}"
        except:
            pass

        try:
            # Check if GPU is available but drivers not loaded
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and 'NVIDIA' in result.stdout:
                return "GPU detected but drivers not loaded"
        except:
            pass

        return None

    def get_optimization_commands(self) -> Tuple[List[str], List[str]]:
        """Get platform-specific optimization commands"""
        optimize_commands = []
        restore_commands = []

        if self.system == "linux":
            optimize_commands = [
                "# Set all CPUs to performance mode",
                "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "",
                "# Disable CPU idle states for maximum responsiveness (optional)",
                "# echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "",
                "# Verify performance mode",
                "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | head -5"
            ]

            restore_commands = [
                "# Restore power-saving mode after processing",
                "echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "",
                "# Or use balanced mode (if available)",
                "# echo schedutil | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
            ]

        elif self.system == "darwin":  # macOS
            optimize_commands = [
                "# Prevent system sleep during processing",
                "sudo pmset -b sleep 0 disksleep 0",
                "sudo pmset -c sleep 0 disksleep 0",
                "",
                "# Set high performance mode",
                "sudo pmset -c womp 0",
                "sudo pmset -c displaysleep 0"
            ]

            restore_commands = [
                "# Restore power management settings",
                "sudo pmset -b sleep 10 disksleep 10",
                "sudo pmset -c sleep 30 disksleep 30",
                "sudo pmset -c displaysleep 10"
            ]

        elif self.system == "windows":
            optimize_commands = [
                "# Set high performance power plan",
                "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
                "",
                "# Disable CPU throttling",
                "powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100",
                "powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100",
                "",
                "# Apply settings",
                "powercfg /setactive SCHEME_CURRENT"
            ]

            restore_commands = [
                "# Restore balanced power plan",
                "powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e"
            ]

        return optimize_commands, restore_commands

    def apply_linux_optimization(self) -> bool:
        """Apply Linux-specific CPU performance optimization"""
        if self.system != "linux":
            return False

        try:
            # Check if we can modify CPU governor
            governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if not governor_path.exists():
                print("⚠️  CPU frequency scaling not available")
                return False

            # Try to set performance mode
            cmd = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True)

            if result.returncode == 0:
                print("🚀 CPU governor set to performance mode")
                return True
            else:
                print("⚠️  Failed to set performance mode (may need sudo)")
                return False

        except Exception as e:
            print(f"⚠️  Error applying optimization: {e}")
            return False

    def print_system_status(self, cpu_workers=None, gpu_workers=None, gpu_acceleration=None):
        """Print comprehensive system performance status"""
        print(f"\n🔍 System Performance Analysis")
        print(f"   • Platform: {platform.system()} {platform.release()}")
        print(f"   • Admin privileges: {'Yes' if self.is_admin else 'No'}")
        print(f"   • CPU cores: {os.cpu_count()}")

        if cpu_workers:
            print(f"   • CPU parallel workers: {cpu_workers}")

        # CPU Governor Status
        governor_info = self.check_cpu_governor()
        if governor_info.get('current'):
            print(f"   • CPU governor: {governor_info['current']}")

        # GPU acceleration status
        if gpu_acceleration is not None:
            print(f"   • GPU acceleration: {'Enabled' if gpu_acceleration else 'Disabled/Not available'}")

        # GPU information check
        gpu_info = self._check_gpu_info()
        if gpu_info:
            gpu_text = f"   • GPU model: {gpu_info['model']}"
            if 'cores' in gpu_info:
                gpu_text += f" ({gpu_info['cores']} cores)"
            print(gpu_text)

        if gpu_workers and gpu_acceleration:
            print(f"   • GPU parallel workers: {gpu_workers}")

        # GPU Governor Status
        gpu_governor_info = self._check_gpu_governor()
        if gpu_governor_info:
            print(f"   • GPU governor: {gpu_governor_info}")

        # Thermal Status
        thermal_info = self.check_thermal_status()
        print(f"   • Thermal monitoring: {thermal_info.get('message', 'Unknown')}")

        if thermal_info.get('thermal_zones'):
            for zone, temp in thermal_info['thermal_zones'].items():
                print(f"     - {zone}: {temp}")

    def print_optimization_guide(self):
        """Print platform-specific optimization commands"""
        optimize_cmds, restore_cmds = self.get_optimization_commands()

        print(f"\n🔧 Performance Optimization Commands ({platform.system()})")
        print(f"{'='*60}")

        print(f"\n📈 BEFORE PROCESSING - Run these commands:")
        for cmd in optimize_cmds:
            if cmd.startswith('#'):
                print(f"\033[92m{cmd}\033[0m")  # Green for comments
            elif cmd.strip():
                print(f"\033[93m{cmd}\033[0m")    # Yellow for commands
            else:
                print()

        print(f"\n📉 AFTER PROCESSING - Restore power saving:")
        for cmd in restore_cmds:
            if cmd.startswith('#'):
                print(f"\033[92m{cmd}\033[0m")  # Green for comments
            elif cmd.strip():
                print(f"\033[93m{cmd}\033[0m")    # Yellow for commands
            else:
                print()

        print(f"\033[0m")  # Reset colors

    def auto_optimize(self) -> bool:
        """Attempt automatic optimization (Linux only for now)"""
        if self.system == "linux":
            print(f"\n🚀 Attempting automatic optimization...")
            return self.apply_linux_optimization()
        else:
            print(f"\n⚠️  Automatic optimization not available for {platform.system()}")
            print(f"Please run the commands shown above manually.")
            return False


def optimize_system_performance(auto_apply: bool = False) -> bool:
    """
    Main function to optimize system performance for video processing

    Args:
        auto_apply: If True, attempt automatic optimization (Linux only)

    Returns:
        bool: True if optimization was applied successfully
    """
    optimizer = PerformanceOptimizer()

    # Print current system status
    optimizer.print_system_status()

    # Show optimization guide
    optimizer.print_optimization_guide()

    # Apply automatic optimization if requested
    if auto_apply:
        return optimizer.auto_optimize()

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='System Performance Optimizer for Video Processing')
    parser.add_argument('--auto', action='store_true', help='Attempt automatic optimization')
    parser.add_argument('--status-only', action='store_true', help='Show status without optimization guide')

    args = parser.parse_args()

    optimizer = PerformanceOptimizer()

    if args.status_only:
        optimizer.print_system_status()
    else:
        optimize_system_performance(auto_apply=args.auto)