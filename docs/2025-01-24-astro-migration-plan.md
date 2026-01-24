# BERTVision Website Migration Plan

**Date:** 2025-01-24
**Goal:** Replace Jekyll website with Astro 5.x static site
**Branch:** `gh-pages`

## Tech Stack

- Astro 5.x (static site generator)
- Tailwind CSS v4 via `@tailwindcss/vite` plugin
- TypeScript (strict mode, no `any` types)
- Vitest (unit tests, co-located with components)
- Playwright (E2E tests: Chromium, Firefox, WebKit)
- gts (Google TypeScript Style for linting/formatting)

## Design Summary

- **Layout:** Single-page with smooth scroll sections
- **Style:** Modern Technical - clean, minimal, generous whitespace
- **Nav:** Centered hero, floating nav appears on scroll
- **Results:** Tabbed interface with collapsible code blocks
- **Diagrams:** Lightbox gallery (click to view full-size)
- **Features:** Animations, copy-to-clipboard, table sorting, search, scroll progress
- **Colors:** Auto-derived accent from hero image, light/dark mode
- **Mobile:** Fully optimized with hamburger menu, swipeable tabs

---

## Phase 1: Project Initialization

### 1.1 Clean up existing Jekyll files
- [ ] Back up current `gh-pages` content (copy to temp location or commit reference)
- [ ] Remove `_config.yml`
- [ ] Remove `index.md`
- [ ] Remove `assets/css/style.scss`
- [ ] Keep `images/` folder (will migrate contents)
- [ ] Keep `paper/` folder (will move to `public/`)
- [ ] Keep `CLAUDE.md` (project documentation)

### 1.2 Initialize Astro project
- [ ] Run `npm create astro@latest` with TypeScript strict template
- [ ] Select empty project template
- [ ] Configure for static output in `astro.config.ts`
- [ ] Set `base` path for GitHub Pages (`/BERTVision/`)
- [ ] Set `site` URL (`https://cbenge509.github.io`)

### 1.3 Configure TypeScript
- [ ] Enable strict mode in `tsconfig.json`
- [ ] Set `noImplicitAny: true`
- [ ] Set `strictNullChecks: true`
- [ ] Configure path aliases (`@components`, `@lib`, etc.)

### 1.4 Install and configure Tailwind CSS v4
- [ ] Install `tailwindcss` and `@tailwindcss/vite`
- [ ] Configure Vite plugin in `astro.config.ts`:
  ```typescript
  import tailwindcss from '@tailwindcss/vite';
  export default defineConfig({
    vite: {
      plugins: [tailwindcss()],
    },
  });
  ```
- [ ] Create `src/styles/global.css` with `@import "tailwindcss"`
- [ ] Import global.css in base layout

### 1.5 Install and configure gts
- [ ] Run `npx gts init`
- [ ] Merge generated configs with Astro's TypeScript config
- [ ] Configure ESLint for Astro files (`.astro` extension)
- [ ] Add lint scripts to `package.json`
- [ ] Verify `npm run lint` works

### 1.6 Install and configure Vitest
- [ ] Install `vitest` and `@testing-library/dom`
- [ ] Create `vitest.config.ts`:
  ```typescript
  import { defineConfig } from 'vitest/config';
  export default defineConfig({
    test: {
      environment: 'jsdom',
      include: ['src/**/__tests__/*.test.ts'],
    },
  });
  ```
- [ ] Add test script to `package.json`
- [ ] Create sample test to verify setup

### 1.7 Install and configure Playwright
- [ ] Run `npm init playwright@latest`
- [ ] Select Chromium, Firefox, WebKit browsers
- [ ] Configure `playwright.config.ts`:
  ```typescript
  export default defineConfig({
    testDir: './tests/e2e',
    webServer: {
      command: 'npm run preview',
      port: 4321,
    },
  });
  ```
- [ ] Install `@axe-core/playwright` for accessibility testing
- [ ] Create sample E2E test to verify setup

### 1.8 Set up project structure
- [ ] Create directory structure:
  ```
  src/
  ├── components/
  │   └── __tests__/
  ├── layouts/
  ├── pages/
  ├── styles/
  ├── content/
  │   └── results/
  ├── lib/
  └── assets/
      └── images/
  tests/
  └── e2e/
  public/
  └── paper/
  ```
- [ ] Verify `npm run dev` starts successfully
- [ ] Verify `npm run build` completes successfully

---

## Phase 2: Asset Migration

### 2.1 Migrate images
- [ ] Copy `images/bertvision.png` → `src/assets/images/`
- [ ] Copy `images/BERTVision_QA_Model.png` → `src/assets/images/`
- [ ] Copy `images/Data_Pipeline_Span_Annotation.png` → `src/assets/images/`
- [ ] Copy `images/BERTVision_development_pipeline.png` → `src/assets/images/`
- [ ] Copy `images/bert_base_vs_bertvision.png` → `src/assets/images/`
- [ ] Extract `BERTVision_hero.jpg` from master branch → `src/assets/images/`
  ```bash
  git show master:images/BERTVision_hero.jpg > src/assets/images/BERTVision_hero.jpg
  ```

### 2.2 Migrate static files
- [ ] Move `paper/BERTVision_2020.pdf` → `public/paper/`
- [ ] Verify PDF accessible at `/BERTVision/paper/BERTVision_2020.pdf`

### 2.3 Set up fonts
- [ ] Add Inter font (Google Fonts or self-hosted)
- [ ] Add JetBrains Mono for code blocks
- [ ] Configure font-face in global.css or via Fontsource

---

## Phase 3: Theme & Styling Foundation

### 3.1 Create color extraction utility
- [ ] Create `src/lib/colors.ts`
- [ ] Implement function to extract dominant color from hero image
- [ ] Generate accent color palette (primary, light, dark variants)
- [ ] Export as CSS custom properties

### 3.2 Configure Tailwind theme
- [ ] Create `src/styles/theme.css` with `@theme` directive
- [ ] Define color tokens:
  - `--color-primary` (from hero image)
  - `--color-primary-light`
  - `--color-primary-dark`
  - Neutral grays
  - Semantic colors (success, warning, error)
- [ ] Define typography scale
- [ ] Define spacing scale
- [ ] Define breakpoints

### 3.3 Set up light/dark mode
- [ ] Create CSS variables for light mode (default)
- [ ] Create CSS variables for dark mode (`.dark` class)
- [ ] Implement theme detection (system preference)
- [ ] Implement theme persistence (localStorage)

### 3.4 Create global styles
- [ ] Base typography styles
- [ ] Link styles with accent color
- [ ] Focus ring styles for accessibility
- [ ] Smooth scroll behavior
- [ ] Reduced motion media query support

### 3.5 Write theme tests
- [ ] Unit test: color extraction function
- [ ] Unit test: theme toggle logic
- [ ] E2E test: dark/light mode switching
- [ ] E2E test: theme persistence across reload

---

## Phase 4: Layout Components

### 4.1 Create BaseLayout
- [ ] Create `src/layouts/BaseLayout.astro`
- [ ] Include HTML boilerplate with proper lang attribute
- [ ] Add meta tags (viewport, description, Open Graph)
- [ ] Import global styles
- [ ] Add slot for page content
- [ ] Include skip-to-content link for accessibility

### 4.2 Create ScrollProgress component
- [ ] Create `src/components/ScrollProgress.astro`
- [ ] Implement fixed position bar at top
- [ ] Add client-side script for scroll tracking
- [ ] Style with accent color
- [ ] Add smooth transitions

### 4.3 Create Nav component
- [ ] Create `src/components/Nav.astro`
- [ ] Implement hidden state (initially off-screen)
- [ ] Add scroll listener to show nav after hero
- [ ] Create section links:
  - Introduction
  - How It Works
  - Data
  - Results
  - Replication
  - Paper
- [ ] Add logo (small version)
- [ ] Add theme toggle button placeholder
- [ ] Add search button placeholder
- [ ] Implement smooth scroll on link click

### 4.4 Create MobileNav component
- [ ] Create `src/components/MobileNav.astro`
- [ ] Implement hamburger button
- [ ] Create slide-out menu panel
- [ ] Add close button and overlay
- [ ] Include all section links
- [ ] Handle body scroll lock when open

### 4.5 Create Section component
- [ ] Create `src/components/Section.astro`
- [ ] Accept props: id, title, variant (light/dark)
- [ ] Implement consistent max-width container
- [ ] Add anchor ID for navigation
- [ ] Add scroll-triggered fade-in animation

### 4.6 Create Footer component
- [ ] Create `src/components/Footer.astro`
- [ ] Add author credits
- [ ] Add year
- [ ] Add links: Paper, GitHub, Repository

### 4.7 Write layout tests
- [ ] Unit test: Nav visibility logic
- [ ] Unit test: Section anchor generation
- [ ] E2E test: Navigation smooth scroll
- [ ] E2E test: Mobile menu open/close
- [ ] E2E test: Responsive breakpoints

---

## Phase 5: Interactive Components

### 5.1 Create ThemeToggle component
- [ ] Create `src/components/ThemeToggle.astro`
- [ ] Add sun/moon icon toggle button
- [ ] Implement theme switching logic
- [ ] Persist preference to localStorage
- [ ] Add aria-label for accessibility

### 5.2 Create SearchModal component
- [ ] Create `src/components/SearchModal.astro`
- [ ] Implement modal overlay
- [ ] Add search input field
- [ ] Create search index from content
- [ ] Implement fuzzy search matching
- [ ] Display results with context preview
- [ ] Handle keyboard navigation (arrows, Enter, Escape)
- [ ] Add Cmd/Ctrl+K shortcut to open
- [ ] Scroll to result on selection

### 5.3 Create CodeBlock component
- [ ] Create `src/components/CodeBlock.astro`
- [ ] Accept props: code, language, showLineNumbers
- [ ] Integrate syntax highlighting (Shiki or Prism)
- [ ] Add copy-to-clipboard button
- [ ] Show "Copied!" feedback
- [ ] Style for both light and dark modes

### 5.4 Create Collapsible component
- [ ] Create `src/components/Collapsible.astro`
- [ ] Accept props: title, defaultOpen
- [ ] Implement expand/collapse animation
- [ ] Add chevron icon indicator
- [ ] Handle keyboard interaction

### 5.5 Create Lightbox component
- [ ] Create `src/components/Lightbox.astro`
- [ ] Implement modal overlay with dark background
- [ ] Display full-size image centered
- [ ] Add close button (X)
- [ ] Close on overlay click
- [ ] Close on Escape key
- [ ] Add navigation arrows for multiple images
- [ ] Implement touch swipe on mobile
- [ ] Trap focus within modal

### 5.6 Create DiagramImage component
- [ ] Create `src/components/DiagramImage.astro`
- [ ] Accept props: src, alt, caption
- [ ] Use Astro's `<Image>` for optimization
- [ ] Add click handler to open Lightbox
- [ ] Style with subtle border/shadow
- [ ] Add hover effect indicating clickability

### 5.7 Write interactive component tests
- [ ] Unit test: CodeBlock copy functionality
- [ ] Unit test: Search indexing and matching
- [ ] Unit test: Collapsible state management
- [ ] Unit test: Lightbox keyboard handling
- [ ] E2E test: Search modal workflow
- [ ] E2E test: Lightbox open/close/navigate
- [ ] E2E test: Copy-to-clipboard

---

## Phase 6: Results Components

### 6.1 Create content schema
- [ ] Create `src/content/config.ts`
- [ ] Define results collection schema:
  ```typescript
  const resultsCollection = defineCollection({
    type: 'data',
    schema: z.object({
      task: z.string(),
      category: z.enum(['glue', 'squad']),
      description: z.string(),
      metrics: z.array(z.object({
        name: z.string(),
        bertBase: z.number(),
        bertLarge: z.number(),
        bvBase: z.number(),
        bvLarge: z.number(),
      })),
      replicationCommands: z.array(z.string()),
    }),
  });
  ```

### 6.2 Create results data files
- [ ] Create `src/content/results/qqp.json`
- [ ] Create `src/content/results/qnli.json`
- [ ] Create `src/content/results/mnli.json`
- [ ] Create `src/content/results/rte.json`
- [ ] Create `src/content/results/sst.json`
- [ ] Create `src/content/results/msr.json`
- [ ] Create `src/content/results/cola.json`
- [ ] Create `src/content/results/stsb.json`
- [ ] Create `src/content/results/squad.json`
- [ ] Validate all data against schema

### 6.3 Create ResultsTable component
- [ ] Create `src/components/ResultsTable.astro`
- [ ] Accept props: metrics array
- [ ] Render table with proper headers
- [ ] Implement column sorting (click to sort)
- [ ] Highlight best value in each row
- [ ] Style for responsive display
- [ ] Add horizontal scroll on mobile

### 6.4 Create ResultsTabs component
- [ ] Create `src/components/ResultsTabs.astro`
- [ ] Implement category tabs: "GLUE Benchmark" | "SQuAD 2.0"
- [ ] Implement sub-tabs for GLUE tasks
- [ ] Add tab panel content areas
- [ ] Implement keyboard navigation (arrow keys)
- [ ] Add proper ARIA attributes
- [ ] Implement swipe gesture on mobile

### 6.5 Create TaskResult component
- [ ] Create `src/components/TaskResult.astro`
- [ ] Accept props: task data object
- [ ] Display task description
- [ ] Include ResultsTable
- [ ] Include Collapsible with replication commands
- [ ] Use CodeBlock for command display

### 6.6 Write results component tests
- [ ] Unit test: Table sorting logic
- [ ] Unit test: Tab state management
- [ ] Unit test: Data validation
- [ ] E2E test: Tab switching
- [ ] E2E test: Table sorting interaction
- [ ] E2E test: Mobile swipe navigation

---

## Phase 7: Page Content Sections

### 7.1 Create Hero section
- [ ] Create `src/components/Hero.astro`
- [ ] Display `BERTVision_hero.jpg` full-width
- [ ] Add gradient overlay at bottom
- [ ] Add project title
- [ ] Add tagline
- [ ] Add CTA buttons: "Read the Paper", "View on GitHub"
- [ ] Add animated scroll indicator (chevron)
- [ ] Set full viewport height

### 7.2 Create Introduction section
- [ ] Create introduction content from `index.md`
- [ ] Add to Section component with id="introduction"
- [ ] Preserve key messaging about parameter-efficiency
- [ ] Style for readability

### 7.3 Create HowItWorks section
- [ ] Create "How It Works" content
- [ ] Add sub-sections:
  - Data Pipeline (with diagram)
  - Model Architecture (with diagram)
  - Development & Training (with diagram)
- [ ] Integrate DiagramImage components
- [ ] Set up Lightbox gallery context

### 7.4 Create Data section
- [ ] Create "NLP Tasks & Data" content
- [ ] Add SQuAD 2.0 description and table
- [ ] Add GLUE Benchmark description and table
- [ ] Style data tables consistently

### 7.5 Create Results section
- [ ] Integrate ResultsTabs component
- [ ] Add hyperparameter information
- [ ] Include summary results image (`bert_base_vs_bertvision.png`)
- [ ] Connect all task result data

### 7.6 Create Replication section
- [ ] Create "Replication of Embeddings" content
- [ ] Organize by GLUE tasks and SQuAD
- [ ] Use Collapsible for each task's commands
- [ ] Use CodeBlock for all bash commands

### 7.7 Create Paper section
- [ ] Create call-to-action for paper
- [ ] Link to PDF in public folder
- [ ] Add paper preview or thumbnail if desired

### 7.8 Assemble index page
- [ ] Create `src/pages/index.astro`
- [ ] Import BaseLayout
- [ ] Import all section components
- [ ] Arrange sections in order:
  1. Hero
  2. Introduction
  3. How It Works
  4. Data
  5. Results
  6. Replication
  7. Paper
  8. Footer
- [ ] Verify all anchor links work

---

## Phase 8: Animations & Polish

### 8.1 Implement scroll animations
- [ ] Create `src/lib/animations.ts` utilities
- [ ] Implement Intersection Observer for scroll detection
- [ ] Add fade-in-up animation to sections
- [ ] Add stagger effect for lists/grids
- [ ] Respect `prefers-reduced-motion`

### 8.2 Implement nav animations
- [ ] Add slide-down animation for floating nav
- [ ] Add smooth transition for nav background
- [ ] Implement active section highlighting in nav

### 8.3 Implement micro-interactions
- [ ] Button hover effects
- [ ] Link underline animations
- [ ] Tab transition effects
- [ ] Copy button feedback animation
- [ ] Lightbox scale-in effect

### 8.4 Performance optimization
- [ ] Optimize images with Astro's `<Image>`
- [ ] Lazy load below-fold images
- [ ] Minimize JavaScript bundle
- [ ] Verify Core Web Vitals scores

### 8.5 Write animation tests
- [ ] E2E test: Scroll animation triggers
- [ ] E2E test: Reduced motion preference

---

## Phase 9: Accessibility & SEO

### 9.1 Accessibility audit
- [ ] Add skip-to-content link
- [ ] Verify all images have alt text
- [ ] Verify color contrast ratios (WCAG AA)
- [ ] Test keyboard navigation through all interactive elements
- [ ] Verify focus indicators are visible
- [ ] Add ARIA labels where needed
- [ ] Test with screen reader

### 9.2 SEO optimization
- [ ] Add comprehensive meta tags
- [ ] Add Open Graph tags for social sharing
- [ ] Add Twitter card meta tags
- [ ] Create `robots.txt` in public folder
- [ ] Add canonical URL

### 9.3 Write accessibility tests
- [ ] E2E test: Keyboard navigation through entire page
- [ ] E2E test: Focus management in modals
- [ ] E2E test: Axe accessibility scan

---

## Phase 10: Testing & Quality Assurance

### 10.1 Complete unit test coverage
- [ ] Run `npm run test` and verify all pass
- [ ] Check coverage report
- [ ] Add tests for any uncovered logic
- [ ] Target: 80%+ coverage on logic-heavy components

### 10.2 Complete E2E test coverage
- [ ] Run `npm run test:e2e` on all browsers
- [ ] Verify Chromium tests pass
- [ ] Verify Firefox tests pass
- [ ] Verify WebKit tests pass
- [ ] Fix any browser-specific issues

### 10.3 Cross-browser testing
- [ ] Test manually in Chrome
- [ ] Test manually in Firefox
- [ ] Test manually in Safari
- [ ] Test manually in Edge
- [ ] Document any known issues

### 10.4 Responsive testing
- [ ] Test on mobile viewport (375px)
- [ ] Test on tablet viewport (768px)
- [ ] Test on desktop viewport (1280px)
- [ ] Test on large desktop (1920px)
- [ ] Verify touch interactions on real device

### 10.5 Linting and formatting
- [ ] Run `npm run lint` and fix all issues
- [ ] Run `npm run fix` for auto-fixable issues
- [ ] Verify no TypeScript `any` types
- [ ] Verify strict mode compliance

---

## Phase 11: Deployment

### 11.1 Create GitHub Actions workflow
- [ ] Create `.github/workflows/deploy.yml`
- [ ] Configure trigger on push to `gh-pages`
- [ ] Add Node.js setup step
- [ ] Add dependency install step
- [ ] Add lint step
- [ ] Add test step
- [ ] Add build step
- [ ] Add deploy to GitHub Pages step

### 11.2 Configure GitHub Pages
- [ ] Verify repository Settings > Pages configured
- [ ] Set source to GitHub Actions
- [ ] Verify custom domain if applicable

### 11.3 Test deployment
- [ ] Push changes to `gh-pages`
- [ ] Monitor GitHub Actions workflow
- [ ] Verify site loads at `https://cbenge509.github.io/BERTVision/`
- [ ] Test all functionality on live site
- [ ] Verify PDF paper link works
- [ ] Verify all images load

### 11.4 Final cleanup
- [ ] Remove any unused files
- [ ] Remove old Jekyll `images/` folder (after migration)
- [ ] Update CLAUDE.md with new website info
- [ ] Create README.md for gh-pages branch (optional)

---

## Appendix: Package Dependencies

### Production Dependencies
```json
{
  "astro": "^5.0.0",
  "@astrojs/check": "^0.9.0"
}
```

### Development Dependencies
```json
{
  "tailwindcss": "^4.0.0",
  "@tailwindcss/vite": "^4.0.0",
  "typescript": "^5.0.0",
  "vitest": "^2.0.0",
  "@testing-library/dom": "^10.0.0",
  "jsdom": "^25.0.0",
  "@playwright/test": "^1.48.0",
  "@axe-core/playwright": "^4.10.0",
  "gts": "^6.0.0"
}
```

---

## Appendix: File Reference

### Key Configuration Files
- `astro.config.ts` - Astro configuration with Tailwind Vite plugin
- `tsconfig.json` - TypeScript strict mode configuration
- `vitest.config.ts` - Vitest test configuration
- `playwright.config.ts` - Playwright E2E configuration
- `.github/workflows/deploy.yml` - GitHub Actions deployment

### Component Inventory
| Component | Type | Tests |
|-----------|------|-------|
| BaseLayout | Layout | - |
| ScrollProgress | UI | Unit + E2E |
| Nav | UI | Unit + E2E |
| MobileNav | UI | E2E |
| Section | UI | Unit |
| Footer | UI | - |
| ThemeToggle | Interactive | Unit + E2E |
| SearchModal | Interactive | Unit + E2E |
| CodeBlock | Interactive | Unit + E2E |
| Collapsible | Interactive | Unit |
| Lightbox | Interactive | Unit + E2E |
| DiagramImage | UI | - |
| ResultsTable | Data | Unit + E2E |
| ResultsTabs | Data | Unit + E2E |
| TaskResult | Data | - |
| Hero | Content | - |

---

## Progress Tracking

Use this section to track overall progress:

- [ ] **Phase 1:** Project Initialization
- [ ] **Phase 2:** Asset Migration
- [ ] **Phase 3:** Theme & Styling Foundation
- [ ] **Phase 4:** Layout Components
- [ ] **Phase 5:** Interactive Components
- [ ] **Phase 6:** Results Components
- [ ] **Phase 7:** Page Content Sections
- [ ] **Phase 8:** Animations & Polish
- [ ] **Phase 9:** Accessibility & SEO
- [ ] **Phase 10:** Testing & Quality Assurance
- [ ] **Phase 11:** Deployment
