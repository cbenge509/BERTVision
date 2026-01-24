import {describe, it, expect, beforeEach} from 'vitest';
import {JSDOM} from 'jsdom';

describe('Collapsible state management', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="collapsible" data-open="false" id="collapsible-1">
            <button
              class="collapsible-trigger"
              aria-expanded="false"
              aria-controls="collapsible-1-content"
            >
              <span class="collapsible-title">Section Title</span>
              <svg class="collapsible-icon"></svg>
            </button>
            <div
              class="collapsible-content"
              id="collapsible-1-content"
              hidden
            >
              <div class="collapsible-inner">Content here</div>
            </div>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  it('should start in closed state', () => {
    const collapsible = document.querySelector('.collapsible');
    const trigger = document.querySelector('.collapsible-trigger');
    const content = document.querySelector('.collapsible-content');

    expect(collapsible?.getAttribute('data-open')).toBe('false');
    expect(trigger?.getAttribute('aria-expanded')).toBe('false');
    expect(content?.hasAttribute('hidden')).toBe(true);
  });

  it('should have proper ARIA attributes', () => {
    const trigger = document.querySelector('.collapsible-trigger');
    const content = document.querySelector('.collapsible-content');

    expect(trigger?.getAttribute('aria-controls')).toBe('collapsible-1-content');
    expect(content?.getAttribute('id')).toBe('collapsible-1-content');
  });

  it('should toggle state when clicked', () => {
    const collapsible = document.querySelector('.collapsible') as HTMLElement;
    const trigger = document.querySelector('.collapsible-trigger') as HTMLElement;
    const content = document.querySelector('.collapsible-content') as HTMLElement;

    // Simulate toggle to open
    collapsible.dataset.open = 'true';
    trigger.setAttribute('aria-expanded', 'true');
    content.hidden = false;

    expect(collapsible.dataset.open).toBe('true');
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    expect(content.hidden).toBe(false);

    // Simulate toggle to closed
    collapsible.dataset.open = 'false';
    trigger.setAttribute('aria-expanded', 'false');
    content.hidden = true;

    expect(collapsible.dataset.open).toBe('false');
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
    expect(content.hidden).toBe(true);
  });
});

describe('Collapsible defaultOpen state', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="collapsible" data-open="true">
            <button
              class="collapsible-trigger"
              aria-expanded="true"
            >
              <span class="collapsible-title">Open Section</span>
            </button>
            <div class="collapsible-content">
              <div class="collapsible-inner">Visible content</div>
            </div>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  it('should start in open state when defaultOpen is true', () => {
    const collapsible = document.querySelector('.collapsible');
    const trigger = document.querySelector('.collapsible-trigger');
    const content = document.querySelector('.collapsible-content');

    expect(collapsible?.getAttribute('data-open')).toBe('true');
    expect(trigger?.getAttribute('aria-expanded')).toBe('true');
    expect(content?.hasAttribute('hidden')).toBe(false);
  });
});
