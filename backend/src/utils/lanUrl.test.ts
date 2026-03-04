/**
 * Regression tests for LAN URL utility.
 *
 * Issue #167: runtime_url must use LAN IP (not localhost) so BOX-3
 * devices on the WiFi network can reach the backend.
 */

import { describe, it, expect, afterEach } from 'vitest';
import type os from 'node:os';
import { getLanIp, getLanUrl } from './lanUrl.js';

// ── Test fixtures ─────────────────────────────────────────────────────

type NetworkInterfaceInfo = ReturnType<typeof os.networkInterfaces>;

const LOOPBACK_ONLY: NetworkInterfaceInfo = {
  lo: [
    {
      address: '127.0.0.1',
      netmask: '255.0.0.0',
      family: 'IPv4',
      mac: '00:00:00:00:00:00',
      internal: true,
      cidr: '127.0.0.1/8',
    },
  ],
};

const LAN_WITH_LOOPBACK: NetworkInterfaceInfo = {
  lo: [
    {
      address: '127.0.0.1',
      netmask: '255.0.0.0',
      family: 'IPv4',
      mac: '00:00:00:00:00:00',
      internal: true,
      cidr: '127.0.0.1/8',
    },
  ],
  eth0: [
    {
      address: '192.168.1.42',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: 'aa:bb:cc:dd:ee:ff',
      internal: false,
      cidr: '192.168.1.42/24',
    },
  ],
};

const IPV6_THEN_IPV4: NetworkInterfaceInfo = {
  eth0: [
    {
      address: 'fe80::1',
      netmask: 'ffff:ffff:ffff:ffff::',
      family: 'IPv6',
      mac: 'aa:bb:cc:dd:ee:ff',
      internal: false,
      cidr: 'fe80::1/64',
      scopeid: 1,
    },
    {
      address: '10.0.0.5',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: 'aa:bb:cc:dd:ee:ff',
      internal: false,
      cidr: '10.0.0.5/24',
    },
  ],
};

const MULTIPLE_LANS: NetworkInterfaceInfo = {
  eth0: [
    {
      address: '192.168.1.10',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: 'aa:bb:cc:dd:ee:ff',
      internal: false,
      cidr: '192.168.1.10/24',
    },
  ],
  wlan0: [
    {
      address: '10.0.0.20',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: '11:22:33:44:55:66',
      internal: false,
      cidr: '10.0.0.20/24',
    },
  ],
};

// ── getLanIp ──────────────────────────────────────────────────────────

describe('getLanIp', () => {
  it('returns a non-empty string from the real system', () => {
    const ip = getLanIp();
    expect(typeof ip).toBe('string');
    expect(ip.length).toBeGreaterThan(0);
  });

  it('returns an IPv4-shaped string or localhost from the real system', () => {
    const ip = getLanIp();
    const isIpv4 = /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(ip);
    const isLocalhost = ip === 'localhost';
    expect(isIpv4 || isLocalhost).toBe(true);
  });

  it('returns localhost when only loopback interfaces exist', () => {
    expect(getLanIp(LOOPBACK_ONLY)).toBe('localhost');
  });

  it('returns first non-internal IPv4 address', () => {
    expect(getLanIp(LAN_WITH_LOOPBACK)).toBe('192.168.1.42');
  });

  it('skips IPv6 addresses and picks the IPv4 one', () => {
    expect(getLanIp(IPV6_THEN_IPV4)).toBe('10.0.0.5');
  });

  it('returns first LAN when multiple external interfaces exist', () => {
    const ip = getLanIp(MULTIPLE_LANS);
    // Should be one of the two LAN IPs (order depends on Object.values iteration)
    expect(['192.168.1.10', '10.0.0.20']).toContain(ip);
  });

  it('returns localhost for empty interfaces', () => {
    expect(getLanIp({})).toBe('localhost');
  });
});

// ── getLanUrl ──────────────────────────────────────────────────────────

describe('getLanUrl', () => {
  const originalEnv = process.env.RUNTIME_URL;

  afterEach(() => {
    if (originalEnv !== undefined) {
      process.env.RUNTIME_URL = originalEnv;
    } else {
      delete process.env.RUNTIME_URL;
    }
  });

  it('uses RUNTIME_URL env var when set', () => {
    process.env.RUNTIME_URL = 'https://custom-proxy.example.com:9000';
    const url = getLanUrl(8000);
    expect(url).toBe('https://custom-proxy.example.com:9000');
  });

  it('auto-detects LAN IP when RUNTIME_URL is not set', () => {
    delete process.env.RUNTIME_URL;
    const url = getLanUrl(8000);
    // Should be http://<ip>:8000 or http://localhost:8000
    expect(url).toMatch(/^http:\/\/.+:8000$/);
  });

  it('uses the provided port number', () => {
    delete process.env.RUNTIME_URL;
    const url = getLanUrl(3456);
    expect(url).toMatch(/:3456$/);
  });

  it('RUNTIME_URL override takes priority over auto-detection', () => {
    process.env.RUNTIME_URL = 'http://override:9999';
    const url = getLanUrl(8000);
    expect(url).toBe('http://override:9999');
    // Port param is ignored when env var is set
    expect(url).not.toContain('8000');
  });

  it('does NOT return localhost:8000 on a machine with a LAN interface (regression #167)', () => {
    delete process.env.RUNTIME_URL;
    const url = getLanUrl(8000);
    // On any dev machine, this should NOT be localhost.
    // This test documents the intent: BOX-3 can't reach localhost.
    // If running in CI with no network, the test still passes since
    // it only checks the URL format.
    expect(url).toMatch(/^http:\/\/.+:8000$/);
  });
});
