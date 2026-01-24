/**
 * Content collection configuration for Astro.
 *
 * Defines the schema for results data used in the Results section.
 */

import {defineCollection, z} from 'astro:content';

/**
 * Schema for a single metric comparison between BERT and BERTVision models.
 */
const metricSchema = z.object({
  /** Name of the metric (e.g., "Accuracy", "F1", "Matthews") */
  name: z.string(),
  /** BERT-base result */
  bertBase: z.number(),
  /** BERT-large result */
  bertLarge: z.number(),
  /** BERTVision-base result */
  bvBase: z.number(),
  /** BERTVision-large result */
  bvLarge: z.number(),
});

/**
 * Schema for results data for each NLP task.
 */
const resultsSchema = z.object({
  /** Task name (e.g., "QQP", "MNLI", "SQuAD") */
  task: z.string(),
  /** Full task name for display */
  taskFullName: z.string(),
  /** Category: GLUE benchmark or SQuAD */
  category: z.enum(['glue', 'squad']),
  /** Brief description of the task */
  description: z.string(),
  /** Dataset URL for reference */
  datasetUrl: z.string().url().optional(),
  /** Array of metric comparisons */
  metrics: z.array(metricSchema),
  /** Commands to replicate the results */
  replicationCommands: z.array(z.string()),
});

/**
 * Results collection for benchmark task results.
 */
const resultsCollection = defineCollection({
  type: 'data',
  schema: resultsSchema,
});

export const collections = {
  results: resultsCollection,
};

export type ResultMetric = z.infer<typeof metricSchema>;
export type ResultData = z.infer<typeof resultsSchema>;
