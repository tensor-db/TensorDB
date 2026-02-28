import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Stub sitemap integration to prevent Starlight from auto-adding the broken one
const noSitemap = { name: '@astrojs/sitemap', hooks: {} };

export default defineConfig({
  site: 'https://tensordb.netlify.app',
  integrations: [
    noSitemap,
    starlight({
      title: 'TensorDB',
      description: 'Documentation for TensorDB — the AI-native bitemporal ledger database',
      social: {
        github: 'https://github.com/tensor-db/TensorDB',
      },
      customCss: ['./src/assets/custom-theme.css'],
      head: [],
      expressiveCode: {
        themes: ['starlight-dark', 'starlight-light'],
      },
      sidebar: [
        {
          label: 'Getting Started',
          collapsed: true,
          items: [
            { label: 'Installation', link: '/getting-started/installation/' },
            { label: 'Quickstart', link: '/getting-started/quickstart/' },
            { label: 'Configuration', link: '/getting-started/configuration/' },
          ],
        },
        { label: 'Roadmap', link: '/roadmap/' },
        {
          label: 'Concepts',
          collapsed: true,
          items: [
            { label: 'Architecture', link: '/concepts/architecture/' },
            { label: 'Data Model', link: '/concepts/data-model/' },
            { label: 'Bitemporal', link: '/concepts/bitemporal/' },
            { label: 'Write Path', link: '/concepts/write-path/' },
            { label: 'Read Path', link: '/concepts/read-path/' },
            { label: 'MVCC', link: '/concepts/mvcc/' },
          ],
        },
        {
          label: 'SQL Reference',
          collapsed: true,
          items: [
            { label: 'Overview', link: '/sql/' },
            { label: 'DDL', link: '/sql/ddl/' },
            { label: 'DML', link: '/sql/dml/' },
            { label: 'Queries', link: '/sql/queries/' },
            { label: 'Temporal Queries', link: '/sql/temporal-queries/' },
            { label: 'EXPLAIN', link: '/sql/explain/' },
            { label: 'Prepared Statements', link: '/sql/prepared-statements/' },
            {
              label: 'Functions',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/sql/functions/' },
                { label: 'String', link: '/sql/functions/string/' },
                { label: 'Numeric', link: '/sql/functions/numeric/' },
                { label: 'Date & Time', link: '/sql/functions/datetime/' },
                { label: 'Aggregate', link: '/sql/functions/aggregate/' },
                { label: 'Window', link: '/sql/functions/window/' },
              ],
            },
          ],
        },
        {
          label: 'Storage Engine',
          collapsed: true,
          items: [
            { label: 'WAL', link: '/storage/wal/' },
            { label: 'Memtable', link: '/storage/memtable/' },
            { label: 'SSTable', link: '/storage/sstable/' },
            { label: 'Compaction', link: '/storage/compaction/' },
            { label: 'Bloom Filters', link: '/storage/bloom-filters/' },
            { label: 'Caching', link: '/storage/caching/' },
            { label: 'Compression', link: '/storage/compression/' },
          ],
        },
        {
          label: 'Engine',
          collapsed: true,
          items: [
            { label: 'Fast Write', link: '/engine/fast-write/' },
            { label: 'Sharding', link: '/engine/sharding/' },
            { label: 'Write Batching', link: '/engine/write-batching/' },
          ],
        },
        {
          label: 'AI Runtime',
          collapsed: true,
          items: [
            { label: 'Overview', link: '/ai/' },
            { label: 'Insights', link: '/ai/insights/' },
            { label: 'Risk Scoring', link: '/ai/risk-scoring/' },
            { label: 'Query Advisor', link: '/ai/query-advisor/' },
            { label: 'Compaction Advisor', link: '/ai/compaction-advisor/' },
            { label: 'Cache Advisor', link: '/ai/cache-advisor/' },
          ],
        },
        {
          label: 'Features',
          collapsed: true,
          items: [
            { label: 'Vector Search', link: '/features/vector-search/' },
            { label: 'Full-Text Search', link: '/features/full-text-search/' },
            { label: 'Time Series', link: '/features/timeseries/' },
            { label: 'Event Sourcing', link: '/features/event-sourcing/' },
            { label: 'Change Feeds', link: '/features/change-feeds/' },
            { label: 'Schema Evolution', link: '/features/schema-evolution/' },
          ],
        },
        {
          label: 'Clustering',
          collapsed: true,
          items: [
            { label: 'Raft Consensus', link: '/cluster/raft/' },
            { label: 'Replication', link: '/cluster/replication/' },
            { label: 'Scaling', link: '/cluster/scaling/' },
          ],
        },
        {
          label: 'Security',
          collapsed: true,
          items: [
            { label: 'Authentication', link: '/security/authentication/' },
            { label: 'Authorization', link: '/security/authorization/' },
          ],
        },
        {
          label: 'Integrations',
          collapsed: true,
          items: [
            { label: 'Python', link: '/integrations/python/' },
            { label: 'Node.js', link: '/integrations/nodejs/' },
            { label: 'CLI', link: '/integrations/cli/' },
          ],
        },
        {
          label: 'Reference',
          collapsed: true,
          items: [
            { label: 'Configuration', link: '/reference/config/' },
            { label: 'API', link: '/reference/api/' },
            { label: 'Errors', link: '/reference/errors/' },
          ],
        },
        {
          label: 'Performance',
          collapsed: true,
          items: [
            { label: 'Benchmarks', link: '/performance/benchmarks/' },
            { label: 'Tuning', link: '/performance/tuning/' },
          ],
        },
      ],
    }),
  ],
});
