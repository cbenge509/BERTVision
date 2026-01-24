/**
 * Unit tests for results data validation.
 */

import {describe, it, expect, beforeAll} from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

interface ResultMetric {
  name: string;
  bertBase: number;
  bertLarge: number;
  bvBase: number;
  bvLarge: number;
}

interface ResultData {
  task: string;
  taskFullName: string;
  category: 'glue' | 'squad';
  description: string;
  datasetUrl?: string;
  metrics: ResultMetric[];
  replicationCommands: string[];
}

const RESULTS_DIR = path.join(process.cwd(), 'src/content/results');

describe('Results Data Validation', () => {
  const resultFiles = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json'));

  describe('All result files exist', () => {
    it('should have all GLUE task files', () => {
      const expectedGlueTasks = ['qqp', 'qnli', 'mnli', 'rte', 'sst', 'msr', 'cola', 'stsb'];
      expectedGlueTasks.forEach(task => {
        expect(resultFiles).toContain(`${task}.json`);
      });
    });

    it('should have SQuAD file', () => {
      expect(resultFiles).toContain('squad.json');
    });
  });

  describe('Schema validation', () => {
    resultFiles.forEach(file => {
      describe(`${file}`, () => {
        let data: ResultData;

        beforeAll(() => {
          const filePath = path.join(RESULTS_DIR, file);
          const content = fs.readFileSync(filePath, 'utf-8');
          data = JSON.parse(content) as ResultData;
        });

        it('should have required fields', () => {
          expect(data.task).toBeDefined();
          expect(typeof data.task).toBe('string');
          expect(data.task.length).toBeGreaterThan(0);

          expect(data.taskFullName).toBeDefined();
          expect(typeof data.taskFullName).toBe('string');
          expect(data.taskFullName.length).toBeGreaterThan(0);

          expect(data.category).toBeDefined();
          expect(['glue', 'squad']).toContain(data.category);

          expect(data.description).toBeDefined();
          expect(typeof data.description).toBe('string');
          expect(data.description.length).toBeGreaterThan(0);

          expect(Array.isArray(data.metrics)).toBe(true);
          expect(data.metrics.length).toBeGreaterThan(0);

          expect(Array.isArray(data.replicationCommands)).toBe(true);
          expect(data.replicationCommands.length).toBeGreaterThan(0);
        });

        it('should have valid metrics', () => {
          data.metrics.forEach(metric => {
            expect(metric.name).toBeDefined();
            expect(typeof metric.name).toBe('string');

            expect(typeof metric.bertBase).toBe('number');
            expect(typeof metric.bertLarge).toBe('number');
            expect(typeof metric.bvBase).toBe('number');
            expect(typeof metric.bvLarge).toBe('number');

            // Values should be between 0 and 1 (as percentages)
            expect(metric.bertBase).toBeGreaterThanOrEqual(0);
            expect(metric.bertBase).toBeLessThanOrEqual(1);
            expect(metric.bertLarge).toBeGreaterThanOrEqual(0);
            expect(metric.bertLarge).toBeLessThanOrEqual(1);
            expect(metric.bvBase).toBeGreaterThanOrEqual(0);
            expect(metric.bvBase).toBeLessThanOrEqual(1);
            expect(metric.bvLarge).toBeGreaterThanOrEqual(0);
            expect(metric.bvLarge).toBeLessThanOrEqual(1);
          });
        });

        it('should have valid replication commands', () => {
          data.replicationCommands.forEach(cmd => {
            expect(typeof cmd).toBe('string');
            expect(cmd.length).toBeGreaterThan(0);
            // Commands should start with python
            expect(cmd.startsWith('python')).toBe(true);
          });
        });

        it('should have valid datasetUrl if present', () => {
          if (data.datasetUrl) {
            expect(data.datasetUrl.startsWith('http')).toBe(true);
          }
        });
      });
    });
  });

  describe('Category consistency', () => {
    it('should have correct categories assigned', () => {
      const glueFiles = ['qqp', 'qnli', 'mnli', 'rte', 'sst', 'msr', 'cola', 'stsb'];

      glueFiles.forEach(task => {
        const filePath = path.join(RESULTS_DIR, `${task}.json`);
        const content = fs.readFileSync(filePath, 'utf-8');
        const data = JSON.parse(content) as ResultData;
        expect(data.category).toBe('glue');
      });

      const squadPath = path.join(RESULTS_DIR, 'squad.json');
      const squadContent = fs.readFileSync(squadPath, 'utf-8');
      const squadData = JSON.parse(squadContent) as ResultData;
      expect(squadData.category).toBe('squad');
    });
  });

  describe('Data integrity', () => {
    it('should have unique task names across all files', () => {
      const tasks: string[] = [];

      resultFiles.forEach(file => {
        const filePath = path.join(RESULTS_DIR, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        const data = JSON.parse(content) as ResultData;
        tasks.push(data.task);
      });

      const uniqueTasks = new Set(tasks);
      expect(uniqueTasks.size).toBe(tasks.length);
    });

    it('should have valid JSON in all files', () => {
      resultFiles.forEach(file => {
        const filePath = path.join(RESULTS_DIR, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        expect(() => JSON.parse(content)).not.toThrow();
      });
    });
  });
});
