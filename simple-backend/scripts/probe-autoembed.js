import 'dotenv/config';
import { MongoClient } from 'mongodb';
const client = new MongoClient(process.env.MONGODB_URI);
await client.connect();
const c = client.db('probe').collection('items');
try {
  await c.createSearchIndex({ name: 'auto_vec', type: 'vectorSearch',
    definition: { fields: [{ type: 'autoEmbed', modality: 'text', path: 'text', model: 'voyage-4-lite' }] } });
  console.log('index created');
} catch (e) { console.log('create:', e.message.slice(0, 200)); }
for (let i = 0; i < 24; i++) {
  const idx = (await c.listSearchIndexes().toArray()).map(x => `${x.name}:${x.status}`);
  console.log(idx.join(' '));
  if (idx.every(s => s.endsWith('READY'))) break;
  await new Promise(r => setTimeout(r, 5000));
}
try {
  const out = await c.aggregate([
    { $vectorSearch: { index: 'auto_vec', path: 'text', query: { text: 'agent picked a day the office is shut' }, limit: 3, numCandidates: 20 } },
    { $project: { _id: 0, text: 1, score: { $meta: 'vectorSearchScore' } } },
  ]).toArray();
  console.log('vectorSearch:', JSON.stringify(out, null, 1));
} catch (e) { console.log('vectorSearch FAIL:', e.message.slice(0, 300)); }
await client.close();
