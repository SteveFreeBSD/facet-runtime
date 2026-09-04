# Isolated FastFlowLM builds

The CachyOS package at `/usr/bin/flm` is the supported default. Upstream
portable releases can be unpacked into a versioned subdirectory here for
comparison without overwriting pacman-managed files.

Example:

```bash
./tooling/fastflowlm/1.0.4/flm version --json
./tooling/fastflowlm/1.0.4/flm validate --json
```

Version directories are intentionally excluded from git.
