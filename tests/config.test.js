import { describe, it, expect, beforeEach } from 'vitest';
import { config } from '../src/config.js';

// Snapshot the defaults once; restore before each test so tests are independent.
const DEFAULTS = { sortMethod: config.sortMethod, cullMode: config.cullMode };

beforeEach(() => {
  Object.assign(config, DEFAULTS);
});

describe('config defaults', () => {
  it('sorts with gpu-bitonic by default', () => {
    expect(config.sortMethod).toBe('gpu-bitonic');
  });

  it('culls nothing by default', () => {
    expect(config.cullMode).toBe('none');
  });
});

describe('config state transitions', () => {
  it('sort method can switch to cpu-radix', () => {
    config.sortMethod = 'cpu-radix';
    expect(config.sortMethod).toBe('cpu-radix');
  });

  it('cull mode can switch to aabb', () => {
    config.cullMode = 'aabb';
    expect(config.cullMode).toBe('aabb');
  });

  it('sort and cull are independent', () => {
    config.sortMethod = 'cpu-radix';
    config.cullMode = 'aabb';
    expect(config.sortMethod).toBe('cpu-radix');
    expect(config.cullMode).toBe('aabb');
  });

  it('can switch sort back to gpu-bitonic', () => {
    config.sortMethod = 'cpu-radix';
    config.sortMethod = 'gpu-bitonic';
    expect(config.sortMethod).toBe('gpu-bitonic');
  });

  it('can switch cull back to none', () => {
    config.cullMode = 'aabb';
    config.cullMode = 'none';
    expect(config.cullMode).toBe('none');
  });
});
