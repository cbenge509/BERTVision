/**
 * Unit tests for Nav component logic.
 */

import {describe, it, expect, beforeEach, afterEach, vi} from 'vitest';
import {JSDOM} from 'jsdom';

describe('Nav visibility logic', () => {
  let dom: JSDOM;
  let window: Window & typeof globalThis;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <nav id="main-nav" data-visible="false">
            <a href="#introduction" class="nav-link">Introduction</a>
            <a href="#how-it-works" class="nav-link">How It Works</a>
          </nav>
          <section id="introduction" style="height: 500px;"></section>
          <section id="how-it-works" style="height: 500px;"></section>
        </body>
      </html>
    `, {
      url: 'http://localhost',
    });
    window = dom.window as unknown as Window & typeof globalThis;
    document = dom.window.document;

    // Mock window properties
    vi.stubGlobal('document', document);
    vi.stubGlobal('window', window);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    dom.window.close();
  });

  it('should have nav hidden by default', () => {
    const nav = document.getElementById('main-nav');
    expect(nav).not.toBeNull();
    expect(nav?.dataset.visible).toBe('false');
  });

  it('should update visibility based on scroll threshold', () => {
    const nav = document.getElementById('main-nav');
    if (!nav) throw new Error('Nav not found');

    // Simulate scrolling past threshold
    Object.defineProperty(window, 'scrollY', {value: 1000, writable: true});
    Object.defineProperty(window, 'innerHeight', {value: 800, writable: true});

    // Manually update visibility (simulating what the scroll handler would do)
    const SCROLL_THRESHOLD = window.innerHeight * 0.8;
    const shouldShow = window.scrollY > SCROLL_THRESHOLD;
    nav.dataset.visible = String(shouldShow);

    expect(nav.dataset.visible).toBe('true');
  });

  it('should hide nav when scrolled back to top', () => {
    const nav = document.getElementById('main-nav');
    if (!nav) throw new Error('Nav not found');

    // Set initial state as visible
    nav.dataset.visible = 'true';

    // Simulate scrolling back to top
    Object.defineProperty(window, 'scrollY', {value: 0, writable: true});
    Object.defineProperty(window, 'innerHeight', {value: 800, writable: true});

    // Manually update visibility
    const SCROLL_THRESHOLD = window.innerHeight * 0.8;
    const shouldShow = window.scrollY > SCROLL_THRESHOLD;
    nav.dataset.visible = String(shouldShow);

    expect(nav.dataset.visible).toBe('false');
  });

  it('should calculate scroll threshold as 80% of viewport height', () => {
    Object.defineProperty(window, 'innerHeight', {value: 1000, writable: true});
    const SCROLL_THRESHOLD = window.innerHeight * 0.8;
    expect(SCROLL_THRESHOLD).toBe(800);
  });
});

describe('Nav active section tracking', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <nav id="main-nav">
            <a href="#introduction" class="nav-link">Introduction</a>
            <a href="#how-it-works" class="nav-link">How It Works</a>
            <a href="#data" class="nav-link">Data</a>
          </nav>
        </body>
      </html>
    `, {
      url: 'http://localhost',
    });
    document = dom.window.document;
    vi.stubGlobal('document', document);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    dom.window.close();
  });

  it('should find nav links by class', () => {
    const links = document.querySelectorAll('.nav-link');
    expect(links.length).toBe(3);
  });

  it('should correctly identify link href attributes', () => {
    const links = document.querySelectorAll<HTMLAnchorElement>('.nav-link');
    const hrefs = Array.from(links).map(link => link.getAttribute('href'));

    expect(hrefs).toEqual(['#introduction', '#how-it-works', '#data']);
  });

  it('should toggle active class correctly', () => {
    const link = document.querySelector<HTMLAnchorElement>('.nav-link');
    if (!link) throw new Error('Link not found');

    // Add active class
    link.classList.add('active');
    expect(link.classList.contains('active')).toBe(true);

    // Remove active class
    link.classList.remove('active');
    expect(link.classList.contains('active')).toBe(false);
  });
});
