import 'dotenv/config';
import { MongoClient } from 'mongodb';

const uri = process.env.MONGODB_URI;
if (!uri) throw new Error('MONGODB_URI missing in .env');

const client = new MongoClient(uri, { serverSelectionTimeoutMS: 10000 });
try {
  await client.connect();
  await client.db('admin').command({ ping: 1 });
  const db = client.db('harness');
  const { insertedId } = await db.collection('smoke').insertOne({ at: new Date() });
  const doc = await db.collection('smoke').findOne({ _id: insertedId });
  await db.collection('smoke').deleteOne({ _id: insertedId });
  console.log('OK: ping + write/read/delete', doc.at.toISOString());
} catch (e) {
  console.error('FAIL:', e.message);
  process.exitCode = 1;
} finally {
  await client.close();
}
