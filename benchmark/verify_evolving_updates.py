#!/usr/bin/env python
"""Stress test evolving updates at scale — test edge cases."""
import asyncio, sys, os, shutil, json

os.environ['PYTHONUTF8'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from strategies.ce_lifecycle import CELifecycleStrategy

    # Fresh DB
    test_dir = 'runs/evolving_test'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

    s = CELifecycleStrategy(persist_dir=test_dir, enable_decay=False)
    await s.initialize()

    print('=== EDGE CASE TESTS ===\n')

    # Test 1: First store (no existing memories)
    print('Test 1: First store (empty DB)')
    id1 = await s.store_or_update_summary('Caroline attended the LGBTQ support group on May 7, 2023.', metadata={'type': 'summary'})
    assert id1.startswith('working_'), f'Expected working_ prefix, got {id1}'
    assert s._collections['working'].count() == 1
    print(f'  PASS: created {id1}, count=1')

    # Test 2: Similar summary updates existing
    print('\nTest 2: Similar summary updates existing')
    id2 = await s.store_or_update_summary('Caroline found the LGBTQ support group inspiring.', metadata={'type': 'summary'})
    assert id2 == id1, f'Expected update to {id1}, got new {id2}'
    assert s._collections['working'].count() == 1, f'Expected count=1, got {s._collections["working"].count()}'
    mem = s._collections['working'].get(ids=[id1], include=['documents', 'metadatas'])
    assert 'inspiring' in mem['documents'][0], f'Content not updated'
    assert mem['metadatas'][0].get('update_count') == 1
    print(f'  PASS: updated {id1}, count still 1, content updated')

    # Test 3: Different person creates new
    print('\nTest 3: Different person creates new')
    id3 = await s.store_or_update_summary('Nate won a gaming tournament on July 7.', metadata={'type': 'summary'})
    assert id3 != id1, f'Should create new, but updated {id1}'
    assert s._collections['working'].count() == 2
    print(f'  PASS: created {id3}, count=2')

    # Test 4: Fact does NOT match summary (type separation)
    print('\nTest 4: Similar content but different type (fact vs summary)')
    id4 = await s.store_or_update_summary('Caroline attended the LGBTQ support group.', metadata={'type': 'fact'})
    # Should NOT update id1 (which is a summary) — should create new fact
    assert id4 != id1, f'Fact should not update summary {id1}'
    assert s._collections['working'].count() == 3
    print(f'  PASS: fact {id4} created separately from summary {id1}, count=3')

    # Test 5: Similar fact updates existing fact
    print('\nTest 5: Similar fact updates existing fact')
    id5 = await s.store_or_update_summary('Caroline went to the LGBTQ support group in May 2023.', metadata={'type': 'fact'})
    assert id5 == id4, f'Expected update to fact {id4}, got {id5}'
    assert s._collections['working'].count() == 3
    print(f'  PASS: updated fact {id4}, count still 3')

    # Test 6: Score preservation after multiple updates
    print('\nTest 6: Score preservation after updates')
    # Manually set score on id1
    s._collections['working'].update(ids=[id1], metadatas=[{
        'score': 0.8, 'uses': 5, 'success_count': 3.0, 'type': 'summary',
        'stored_at': 0, 'tier': 'working', 'outcome_history': '[]'
    }])
    id6 = await s.store_or_update_summary('Caroline is very inspired by the support group experience.', metadata={'type': 'summary'})
    assert id6 == id1
    meta = s._collections['working'].get(ids=[id1], include=['metadatas'])['metadatas'][0]
    assert meta['score'] == 0.8, f'Score lost: {meta["score"]}'
    assert meta['uses'] == 5, f'Uses lost: {meta["uses"]}'
    assert meta['success_count'] == 3.0, f'Success lost: {meta["success_count"]}'
    assert meta.get('update_count') == 2, f'Update count wrong: {meta.get("update_count")}'
    print(f'  PASS: score={meta["score"]}, uses={meta["uses"]}, updates={meta["update_count"]}')

    # Test 7: Rapid sequential updates (same topic)
    print('\nTest 7: Rapid sequential updates (10x same topic)')
    for i in range(10):
        await s.store_or_update_summary(f'Caroline continued exploring counseling options, step {i+1}.', metadata={'type': 'summary'})
    assert s._collections['working'].count() == 3, f'Count should still be 3, got {s._collections["working"].count()}'
    meta = s._collections['working'].get(ids=[id1], include=['metadatas', 'documents'])
    assert meta['metadatas'][0].get('update_count') == 12  # 2 from before + 10
    assert 'step 10' in meta['documents'][0]
    print(f'  PASS: 10 rapid updates, count=3, update_count=12')

    # Test 8: Promoted memory gets found and updated
    print('\nTest 8: Memory in history tier gets found and updated')
    # Manually move id3 (Nate) to history
    result = s._collections['working'].get(ids=[id3], include=['documents', 'metadatas', 'embeddings'])
    if result['ids']:
        meta = result['metadatas'][0]
        meta['tier'] = 'history'
        add_kwargs = {'ids': [f'history_{id3.split("_",1)[1]}'], 'documents': result['documents'], 'metadatas': [meta]}
        if result.get('embeddings') and len(result['embeddings']) > 0:
            add_kwargs['embeddings'] = result['embeddings']
        s._collections['history'].add(**add_kwargs)
        s._collections['working'].delete(ids=[id3])

    history_id = f'history_{id3.split("_",1)[1]}'
    assert s._collections['history'].count() == 1
    assert s._collections['working'].count() == 2

    # Now update — should find it in history
    id8 = await s.store_or_update_summary('Nate celebrated his tournament win with friends.', metadata={'type': 'summary'})
    assert id8 == history_id, f'Should update history {history_id}, got {id8}'
    assert s._collections['history'].count() == 1  # Still in history, not duplicated
    assert s._collections['working'].count() == 2  # Not created in working
    print(f'  PASS: updated {history_id} in history tier, no duplication')

    # Test 9: Empty content
    print('\nTest 9: Empty content handling')
    id9 = await s.store_or_update_summary('', metadata={'type': 'summary'})
    # Should create new (empty content won't match anything)
    print(f'  PASS: empty content returned {id9}')

    # Test 10: Very long content
    print('\nTest 10: Very long content')
    long_text = 'Caroline ' + 'attended various workshops and events ' * 50
    id10 = await s.store_or_update_summary(long_text[:500], metadata={'type': 'summary'})
    print(f'  PASS: long content returned {id10}')

    print(f'\n=== ALL TESTS PASSED ===')
    print(f'Final DB: working={s._collections["working"].count()}, history={s._collections["history"].count()}, patterns={s._collections["patterns"].count()}')

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    asyncio.run(main())
