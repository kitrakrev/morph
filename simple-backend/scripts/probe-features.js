import 'dotenv/config';
import { MongoClient } from 'mongodb';

const client = new MongoClient(process.env.MONGODB_URI);
const db = client.db('probe');
const r = {};
const t = async (name, fn) => {
  try { r[name] = 'OK ' + ((await fn()) ?? ''); }
  catch (e) { r[name] = 'FAIL ' + (e.codeName || '') + ' ' + e.message.slice(0, 160); }
};

await client.connect();
const c = db.collection('items');
await c.drop().catch(() => {});
await c.insertMany([
  { text: 'agent booked appointment on a Sunday when clinic closed', n: 1 },
  { text: 'refund issued twice after retry', n: 2 },
  { text: 'scheduled on a closed date', n: 3 },
]);

await t('serverVersion', async () => (await db.admin().serverInfo()).version);
await t('clusterTier', async () => {
  const h = await db.admin().command({ hello: 1 });
  return `setName=${h.setName} replicaSet=${!!h.setName}`;
});
await t('uniqueIndex', async () => {
  await c.createIndex({ n: 1 }, { unique: true });
  try { await c.insertOne({ n: 1 }); return 'dup NOT blocked'; }
  catch (e) { return e.code === 11000 ? 'dup blocked' : e.message; }
});
await t('transactions', async () => {
  const s = client.startSession();
  try {
    await s.withTransaction(async () => {
      await c.insertOne({ n: 10 }, { session: s });
      await db.collection('cp').insertOne({ n: 10 }, { session: s });
    });
  } finally { await s.endSession(); }
  return 'committed';
});
await t('changeStreams', async () => {
  const cs = c.watch();
  const p = cs.next();
  setTimeout(() => c.insertOne({ n: 99 }), 500);
  const ev = await Promise.race([p, new Promise((_, j) => setTimeout(() => j(new Error('timeout')), 8000))]);
  await cs.close();
  return ev.operationType;
});
await t('listSearchIndexes', async () => JSON.stringify((await c.listSearchIndexes().toArray()).map(i => i.name)));
await t('createSearchIndex(text)', async () =>
  c.createSearchIndex({ name: 'text_idx', definition: { mappings: { dynamic: true } } }));
await t('createVectorIndex(autoEmbed)', async () =>
  c.createSearchIndex({
    name: 'auto_vec', type: 'vectorSearch',
    definition: { fields: [{ type: 'autoEmbed', modality: 'text', path: 'text', model: 'voyage-3.5-lite' }] },
  }));
await t('createVectorIndex(classic)', async () =>
  c.createSearchIndex({
    name: 'vec', type: 'vectorSearch',
    definition: { fields: [{ type: 'vector', path: 'emb', numDimensions: 1024, similarity: 'cosine' }] },
  }));
await t('$rankFusion parse', async () => {
  const out = await c.aggregate([{ $rankFusion: { input: { pipelines: {
    a: [{ $match: { n: { $gte: 1 } } }, { $sort: { n: 1 } }],
    b: [{ $match: { n: { $lte: 3 } } }, { $sort: { n: -1 } }],
  } } } }]).toArray();
  return `${out.length} docs`;
});

console.log(JSON.stringify(r, null, 2));
await client.close();
