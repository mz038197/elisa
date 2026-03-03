import { describe, it, expect } from 'vitest';
import { resolveDeployOrder } from '../../services/phases/deployOrder.js';

describe('resolveDeployOrder', () => {
  it('sorts cloud before flash device that requires cloud_url', () => {
    const devices = [
      { pluginId: 'gateway', instanceId: 'g1', fields: {} },
      { pluginId: 'cloud', instanceId: 'c1', fields: {} },
    ];
    const manifests = new Map([
      ['cloud', { deploy: { method: 'cloud', provides: ['cloud_url'], requires: [] } }],
      ['gateway', { deploy: { method: 'flash', provides: [], requires: ['cloud_url'], flash: {} } }],
    ]);
    const order = resolveDeployOrder(devices, manifests as any);
    const ids = order.map(d => d.pluginId);
    expect(ids.indexOf('cloud')).toBeLessThan(ids.indexOf('gateway'));
  });

  it('keeps independent devices in input order', () => {
    const devices = [
      { pluginId: 'sensor', instanceId: 's1', fields: {} },
      { pluginId: 'blink', instanceId: 'b1', fields: {} },
    ];
    const manifests = new Map([
      ['sensor', { deploy: { method: 'flash', provides: [], requires: [], flash: {} } }],
      ['blink', { deploy: { method: 'flash', provides: [], requires: [], flash: {} } }],
    ]);
    const order = resolveDeployOrder(devices, manifests as any);
    expect(order.map(d => d.pluginId)).toEqual(['sensor', 'blink']);
  });

  it('throws on circular dependency', () => {
    const devices = [
      { pluginId: 'a', instanceId: 'a1', fields: {} },
      { pluginId: 'b', instanceId: 'b1', fields: {} },
    ];
    const manifests = new Map([
      ['a', { deploy: { method: 'flash', provides: ['x'], requires: ['y'], flash: {} } }],
      ['b', { deploy: { method: 'flash', provides: ['y'], requires: ['x'], flash: {} } }],
    ]);
    expect(() => resolveDeployOrder(devices, manifests as any)).toThrow(/cycle/i);
  });

  it('returns empty array for empty input', () => {
    expect(resolveDeployOrder([], new Map() as any)).toEqual([]);
  });

  it('handles multiple instances of the same pluginId', () => {
    const devices = [
      { pluginId: 'box3', instanceId: 'b1', fields: {} },
      { pluginId: 'box3', instanceId: 'b2', fields: {} },
    ];
    const manifests = new Map([
      ['box3', { deploy: { method: 'flash', provides: ['endpoint'], requires: [], flash: {} } }],
    ]);
    const order = resolveDeployOrder(devices, manifests as any);
    expect(order).toHaveLength(2);
    expect(order.map(d => d.instanceId)).toEqual(['b1', 'b2']);
  });

  it('sorts with mixed duplicate and dependent devices', () => {
    const devices = [
      { pluginId: 'gateway', instanceId: 'g1', fields: {} },
      { pluginId: 'cloud', instanceId: 'c1', fields: {} },
      { pluginId: 'gateway', instanceId: 'g2', fields: {} },
    ];
    const manifests = new Map([
      ['cloud', { deploy: { method: 'cloud', provides: ['cloud_url'], requires: [] } }],
      ['gateway', { deploy: { method: 'flash', provides: [], requires: ['cloud_url'], flash: {} } }],
    ]);
    const order = resolveDeployOrder(devices, manifests as any);
    const ids = order.map(d => d.pluginId);
    // Cloud must come before both gateways
    expect(ids[0]).toBe('cloud');
    expect(order).toHaveLength(3);
  });
});
