/**
 * Animation utilities for BERTVision website.
 *
 * Provides centralized animation helpers including:
 * - Reduced motion preference detection
 * - IntersectionObserver scroll animations
 * - Stagger effects for lists and grids
 * - Animation timing utilities
 */

/** No-operation cleanup function */
function noop(): void {
  // Intentional no-op for cleanup functions when no cleanup is needed
}

/**
 * Check if user prefers reduced motion.
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Listen for reduced motion preference changes.
 */
export function onReducedMotionChange(
  callback: (prefersReduced: boolean) => void
): () => void {
  if (typeof window === 'undefined') return noop;

  const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const handler = (e: MediaQueryListEvent): void => callback(e.matches);

  mediaQuery.addEventListener('change', handler);
  return () => mediaQuery.removeEventListener('change', handler);
}

/**
 * Animation timing presets in milliseconds.
 */
export const TIMING = {
  /** Fast interactions (buttons, links) */
  fast: 150,
  /** Standard UI transitions */
  normal: 200,
  /** Moderate transitions (panels, modals) */
  moderate: 300,
  /** Slow reveals (sections, content) */
  slow: 500,
  /** Section fade-in animations */
  section: 600,
} as const;

/**
 * Easing function presets.
 */
export const EASING = {
  /** Standard ease for general use */
  default: 'ease',
  /** Acceleration */
  in: 'ease-in',
  /** Deceleration */
  out: 'ease-out',
  /** Smooth start and end */
  inOut: 'ease-in-out',
  /** Cubic bezier for snappy animations */
  snappy: 'cubic-bezier(0.4, 0, 0.2, 1)',
} as const;

/**
 * Options for scroll-triggered animations.
 */
export interface ScrollAnimationOptions {
  /** CSS class to add when element is visible */
  animateClass?: string;
  /** IntersectionObserver root margin */
  rootMargin?: string;
  /** Visibility threshold (0-1) */
  threshold?: number;
  /** Whether to animate only once */
  once?: boolean;
}

const defaultScrollOptions: Required<ScrollAnimationOptions> = {
  animateClass: 'animate-in',
  rootMargin: '-50px 0px',
  threshold: 0.1,
  once: true,
};

/**
 * Initialize scroll-triggered animations for elements.
 *
 * @param selector - CSS selector for target elements
 * @param options - Animation options
 * @returns Cleanup function to disconnect observer
 */
export function initScrollAnimations(
  selector: string,
  options: ScrollAnimationOptions = {}
): () => void {
  if (typeof window === 'undefined') return noop;

  const elements = document.querySelectorAll<HTMLElement>(selector);
  if (elements.length === 0) return noop;

  const opts = {...defaultScrollOptions, ...options};

  // If reduced motion, show all elements immediately
  if (prefersReducedMotion()) {
    elements.forEach(el => el.classList.add(opts.animateClass));
    return noop;
  }

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add(opts.animateClass);
          if (opts.once) {
            observer.unobserve(entry.target);
          }
        } else if (!opts.once) {
          entry.target.classList.remove(opts.animateClass);
        }
      });
    },
    {
      root: null,
      rootMargin: opts.rootMargin,
      threshold: opts.threshold,
    }
  );

  elements.forEach(el => observer.observe(el));

  return () => observer.disconnect();
}

/**
 * Options for stagger animations.
 */
export interface StaggerOptions {
  /** Base delay in ms before first item animates */
  baseDelay?: number;
  /** Delay between each item in ms */
  staggerDelay?: number;
  /** CSS class to add when animating */
  animateClass?: string;
  /** Animation duration in ms */
  duration?: number;
}

const defaultStaggerOptions: Required<StaggerOptions> = {
  baseDelay: 0,
  staggerDelay: 50,
  animateClass: 'stagger-in',
  duration: TIMING.slow,
};

/**
 * Apply stagger animation to a list of elements.
 *
 * @param elements - Elements to animate
 * @param options - Stagger options
 */
export function applyStagger(
  elements: NodeListOf<HTMLElement> | HTMLElement[],
  options: StaggerOptions = {}
): void {
  const opts = {...defaultStaggerOptions, ...options};

  // If reduced motion, show all immediately
  if (prefersReducedMotion()) {
    Array.from(elements).forEach(el => {
      el.classList.add(opts.animateClass);
      el.style.transitionDelay = '0ms';
    });
    return;
  }

  Array.from(elements).forEach((el, index) => {
    const delay = opts.baseDelay + index * opts.staggerDelay;
    el.style.transitionDelay = `${delay}ms`;
    el.style.transitionDuration = `${opts.duration}ms`;

    // Use requestAnimationFrame to ensure styles are applied before class
    requestAnimationFrame(() => {
      el.classList.add(opts.animateClass);
    });
  });
}

/**
 * Initialize stagger animations on scroll for container elements.
 *
 * @param containerSelector - CSS selector for containers
 * @param childSelector - CSS selector for children within containers
 * @param options - Stagger and scroll options
 * @returns Cleanup function
 */
export function initStaggerOnScroll(
  containerSelector: string,
  childSelector: string,
  options: StaggerOptions & ScrollAnimationOptions = {}
): () => void {
  if (typeof window === 'undefined') return noop;

  const containers = document.querySelectorAll<HTMLElement>(containerSelector);
  if (containers.length === 0) return noop;

  const scrollOpts: ScrollAnimationOptions = {
    rootMargin: options.rootMargin ?? '-50px 0px',
    threshold: options.threshold ?? 0.1,
    once: options.once ?? true,
  };

  const animateClass = options.animateClass ?? 'stagger-in';
  const staggerOpts: Required<StaggerOptions> = {
    baseDelay: options.baseDelay ?? 0,
    staggerDelay: options.staggerDelay ?? 50,
    animateClass,
    duration: options.duration ?? TIMING.slow,
  };

  // If reduced motion, show all immediately
  if (prefersReducedMotion()) {
    containers.forEach(container => {
      const children = container.querySelectorAll<HTMLElement>(childSelector);
      children.forEach(child => {
        child.classList.add(animateClass);
        child.style.opacity = '1';
        child.style.transform = 'none';
      });
    });
    return noop;
  }

  const observerOptions: IntersectionObserverInit = {
    root: null,
    rootMargin: scrollOpts.rootMargin ?? '-50px 0px',
    threshold: scrollOpts.threshold ?? 0.1,
  };

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const container = entry.target as HTMLElement;
        const children =
          container.querySelectorAll<HTMLElement>(childSelector);
        applyStagger(children, staggerOpts);

        if (scrollOpts.once) {
          observer.unobserve(container);
        }
      }
    });
  }, observerOptions);

  containers.forEach(container => observer.observe(container));

  return () => observer.disconnect();
}

/**
 * CSS custom properties for animations.
 * Can be injected into document root.
 */
export const ANIMATION_CSS_VARS = `
  --animation-timing-fast: ${TIMING.fast}ms;
  --animation-timing-normal: ${TIMING.normal}ms;
  --animation-timing-moderate: ${TIMING.moderate}ms;
  --animation-timing-slow: ${TIMING.slow}ms;
  --animation-timing-section: ${TIMING.section}ms;
  --animation-easing-default: ${EASING.default};
  --animation-easing-in: ${EASING.in};
  --animation-easing-out: ${EASING.out};
  --animation-easing-in-out: ${EASING.inOut};
  --animation-easing-snappy: ${EASING.snappy};
`;

/**
 * Generate stagger animation CSS for a given number of items.
 *
 * @param count - Number of items
 * @param baseDelay - Base delay in ms
 * @param staggerDelay - Delay between items in ms
 * @returns CSS string with nth-child delays
 */
export function generateStaggerCSS(
  count: number,
  baseDelay = 0,
  staggerDelay = 50
): string {
  let css = '';
  for (let i = 1; i <= count; i++) {
    const delay = baseDelay + (i - 1) * staggerDelay;
    css += `.stagger-item:nth-child(${i}) { transition-delay: ${delay}ms; }\n`;
  }
  return css;
}

/**
 * Type for animation state callbacks.
 */
export type AnimationStateCallback = (element: HTMLElement) => void;

/**
 * Options for animated element controller.
 */
export interface AnimatedElementOptions {
  /** Element to animate */
  element: HTMLElement;
  /** Callback when animation starts */
  onStart?: AnimationStateCallback;
  /** Callback when animation ends */
  onEnd?: AnimationStateCallback;
  /** CSS class that triggers animation */
  animateClass?: string;
}

/**
 * Create an animated element controller with lifecycle callbacks.
 */
export function createAnimatedElement(
  options: AnimatedElementOptions
): {
  trigger: () => void;
  reset: () => void;
  isAnimating: () => boolean;
} {
  const {
    element,
    onStart,
    onEnd,
    animateClass = 'animate-in',
  } = options;

  let animating = false;

  const handleAnimationEnd = (): void => {
    animating = false;
    onEnd?.(element);
  };

  element.addEventListener('animationend', handleAnimationEnd);
  element.addEventListener('transitionend', handleAnimationEnd);

  return {
    trigger: () => {
      if (prefersReducedMotion()) {
        element.classList.add(animateClass);
        onStart?.(element);
        onEnd?.(element);
        return;
      }

      animating = true;
      onStart?.(element);
      element.classList.add(animateClass);
    },
    reset: () => {
      animating = false;
      element.classList.remove(animateClass);
    },
    isAnimating: () => animating,
  };
}
