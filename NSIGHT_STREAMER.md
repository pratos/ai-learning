# NVIDIA Nsight Streamer Setup

This document explains how to use the NVIDIA Nsight Streamer integration for web-based profiling analysis.

## Overview

The NVIDIA Nsight Streamer provides a web-based interface for analyzing NVIDIA Nsight Systems profiling data directly in your browser. This is particularly useful for:

- **Remote Analysis**: Analyze large profiling files without transferring them to local machines
- **Enhanced Performance**: Leverage server GPU resources for faster analysis
- **Team Collaboration**: Share profiling analysis sessions via web interface
- **Security Compliance**: Keep sensitive profiling data on secure servers

## Quick Start

### 1. Generate Profiling Data

First, run your training with NVTX profiling to generate `.nsys-rep` files:

```bash
./bin/run train-nvtx 5 256 gpu_experiment 4
```

This will create profiling files in `./profiling_output/` directory.

### 2. Start Nsight Streamer

```bash
./bin/run nsight
```

### 3. Access Web Interface

Open your browser and navigate to:
- **URL**: http://localhost:8080
- **Username**: nvidia
- **Password**: nvidia

The interface will automatically detect `.nsys-rep` files from your `./profiling_output/` directory.

### 4. Stop Nsight Streamer

```bash
./bin/run nsight-stop
```

## Configuration

### Default Settings

The Nsight Streamer is configured with the following defaults:

- **HTTP Port**: 8080
- **TURN Port**: 3478 (for WebRTC)
- **Username**: nvidia
- **Password**: nvidia
- **Max Resolution**: 1920x1080
- **Resize Enabled**: Yes

### Custom Configuration

You can modify the settings in `docker-compose.yml` under the `nsight-streamer` service:

```yaml
environment:
  - TURN_PORT=3478
  - ENABLE_RESIZE=true
  - MAX_RESOLUTION=1920x1080
  - WEB_USERNAME=your_username
  - WEB_PASSWORD=your_password
```

## Browser Compatibility

### Chrome (Recommended)
- Full feature support
- For clipboard functionality, enable: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
- Add `http://localhost:8080` to the list

### Firefox
- Full feature support
- If you encounter "Connection failed" errors:
  1. Navigate to `about:config`
  2. Search for `media.peerconnection.ice.loopback`
  3. Set the value to `true`

### Safari
- For AV1 codec support:
  1. Go to Safari > Settings > Advanced
  2. Check "Show features for web developers"
  3. Go to "Feature Flags" and enable "WebRTC AV1 codec"
  4. Restart Safari

## Hardware Acceleration

The Nsight Streamer automatically detects and uses NVIDIA GPU hardware acceleration if available:

- **Requirements**: NVIDIA GPU with AV1 encoding support (Ada Lovelace or newer)
- **Benefits**: Improved streaming performance and responsiveness
- **Automatic Detection**: Enabled by default when compatible GPU is detected

## File Structure

```
./profiling_output/          # Profiling reports (auto-mounted)
├── training_profile_5ep.nsys-rep
├── gpu_experiment_3ep.nsys-rep
└── ...

./data/                      # Training data (mounted for reference)
./logs/                      # Training logs (mounted for reference)
```

## Troubleshooting

### Connection Issues

1. **Port Conflicts**: Ensure ports 8080 and 3478 are not in use
2. **Firewall**: Check that ports are open in your firewall
3. **Browser Cache**: Clear browser cache and cookies

### Performance Issues

1. **GPU Acceleration**: Ensure NVIDIA drivers are installed on host
2. **Network**: Use wired connection for better stability
3. **Resolution**: Lower `MAX_RESOLUTION` if experiencing lag

### GUI Scaling

If the GUI appears too small on high-DPI displays:
1. In Nsight Systems, go to File > Exit
2. The application will restart with proper scaling

## Workflow Integration

### Typical Analysis Workflow

1. **Generate Profiling Data**:
   ```bash
   ./bin/run train-nvtx 5 256 experiment_name
   ```

2. **Start Analysis**:
   ```bash
   ./bin/run nsight
   ```

3. **Analyze in Browser**:
   - Open http://localhost:8080
   - Navigate to automatically detected reports
   - Use NVTX markers to analyze training epochs

4. **Stop When Done**:
   ```bash
   ./bin/run nsight-stop
   ```

### Multiple Analysis Sessions

The Nsight Streamer can handle multiple profiling files simultaneously. Each training run with different job IDs will create separate reports that can be analyzed independently.

## Security Notes

- Default credentials are `nvidia/nvidia` - change these for production use
- The web interface is accessible from any browser that can reach the host
- Consider using SSH tunneling for remote access: `ssh -L 8080:localhost:8080 user@host`

## Links

- [NVIDIA Nsight Streamer Documentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [NGC Container Registry](https://ngc.nvidia.com/) 