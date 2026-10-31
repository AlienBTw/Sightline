import json,sys
r=json.load(open(sys.argv[1]))
print(r['sizes'], 'pos_weight', [round(x,1) for x in r['pos_weight']], 'min', round(r['minutes'],1))
for f,d in r['test_by_source']['voxel51'].items():
    t=r['test'].get(f,{})
    print(f, {k:round(v,2) if isinstance(v,float) else v for k,v in d.items()}, 'thr/F1/P/R', [round(t.get(k,0),2) for k in ('threshold','f1','precision','recall')])
