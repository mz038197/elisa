/**
 * LAN IP detection utility for runtime URL generation.
 *
 * BOX-3 (ESP32) devices connect over WiFi and cannot resolve "localhost".
 * This utility detects the host machine's LAN IP so the runtime_url
 * returned during agent provisioning is reachable from the device.
 *
 * Supports a RUNTIME_URL env var override for manual configuration
 * (e.g. when behind a reverse proxy or in a Docker container).
 */

import os from 'node:os';

/**
 * Return the first non-internal IPv4 address from the given interfaces,
 * or 'localhost' if none found.
 *
 * @param interfaces - Override for testing. Defaults to os.networkInterfaces().
 */
export function getLanIp(
  interfaces?: ReturnType<typeof os.networkInterfaces>,
): string {
  const ifaces = interfaces ?? os.networkInterfaces();
  for (const iface of Object.values(ifaces)) {
    if (!iface) continue;
    for (const addr of iface) {
      if (addr.family === 'IPv4' && !addr.internal) return addr.address;
    }
  }
  return 'localhost';
}

/**
 * Build the runtime URL that ESP32 devices will use to reach this server.
 *
 * Resolution order:
 *   1. RUNTIME_URL env var (explicit override)
 *   2. http://<LAN_IP>:<port>  (auto-detected)
 */
export function getLanUrl(port: number): string {
  if (process.env.RUNTIME_URL) {
    return process.env.RUNTIME_URL;
  }
  return `http://${getLanIp()}:${port}`;
}
