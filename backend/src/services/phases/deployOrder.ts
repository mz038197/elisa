/** Resolve device deploy order using provides/requires DAG. */

import type { DeviceManifest } from '../../utils/deviceManifestSchema.js';

export interface DeviceInstance {
  pluginId: string;
  instanceId: string;
  fields: Record<string, unknown>;
}

/**
 * Sort devices into deploy order using provides/requires dependencies.
 * Devices that provide keys required by other devices are deployed first.
 * Multiple instances of the same pluginId are kept together in sorted order.
 * Throws if a circular dependency is detected.
 */
export function resolveDeployOrder(
  devices: DeviceInstance[],
  manifests: Map<string, DeviceManifest>,
): DeviceInstance[] {
  if (devices.length === 0) return [];

  // Deduplicate by pluginId for the topological sort (the DAG is
  // between device *types*, not instances). Multiple instances of the
  // same type share the same provides/requires and deploy together.
  const uniquePluginIds = [...new Set(devices.map(d => d.pluginId))];

  // Build a map from provided key -> pluginId
  const providerOf = new Map<string, string>();
  for (const pluginId of uniquePluginIds) {
    const manifest = manifests.get(pluginId);
    if (!manifest) continue;
    for (const key of manifest.deploy.provides) {
      providerOf.set(key, pluginId);
    }
  }

  // Build dependency graph: pluginId -> set of pluginIds it depends on
  const deps = new Map<string, Set<string>>();
  for (const pluginId of uniquePluginIds) {
    const manifest = manifests.get(pluginId);
    if (!manifest) {
      deps.set(pluginId, new Set());
      continue;
    }
    const required = new Set<string>();
    for (const key of manifest.deploy.requires) {
      const provider = providerOf.get(key);
      if (provider && provider !== pluginId) {
        required.add(provider);
      }
    }
    deps.set(pluginId, required);
  }

  // Kahn's topological sort
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const [node, nodeDeps] of deps) {
    if (!inDegree.has(node)) inDegree.set(node, 0);
    if (!adjacency.has(node)) adjacency.set(node, []);
    for (const dep of nodeDeps) {
      if (!inDegree.has(dep)) inDegree.set(dep, 0);
      if (!adjacency.has(dep)) adjacency.set(dep, []);
      adjacency.get(dep)!.push(node);
      inDegree.set(node, (inDegree.get(node) ?? 0) + 1);
    }
  }

  // Seed queue with zero-indegree nodes in original order
  const queue: string[] = [];
  for (const id of uniquePluginIds) {
    if ((inDegree.get(id) ?? 0) === 0) {
      queue.push(id);
    }
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const node = queue.shift()!;
    sorted.push(node);
    for (const neighbor of adjacency.get(node) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  if (sorted.length !== inDegree.size) {
    throw new Error('Circular dependency cycle detected in device deploy order');
  }

  // Expand sorted pluginIds back to all device instances, preserving
  // input order among instances of the same type.
  const instancesByPlugin = new Map<string, DeviceInstance[]>();
  for (const device of devices) {
    let list = instancesByPlugin.get(device.pluginId);
    if (!list) {
      list = [];
      instancesByPlugin.set(device.pluginId, list);
    }
    list.push(device);
  }

  const result: DeviceInstance[] = [];
  for (const pluginId of sorted) {
    const instances = instancesByPlugin.get(pluginId);
    if (instances) result.push(...instances);
  }
  return result;
}
