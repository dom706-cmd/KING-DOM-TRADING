import io, pickle

class AnyShim:
    def __init__(self, *a, **k):
        self.a=a; self.k=k; self.state=None
    def __call__(self, *a, **k):
        return AnyShim(*a, **k)
    def __setstate__(self, state):
        # accept any state shape
        self.state = state
    def __getstate__(self):
        return self.state

class TolerantTime(AnyShim):
    # specifically handle tuple length 4 or 5
    def __setstate__(self, state):
        self.state = state

class FixingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # handle datetime time fold mismatch across builds
        if module in ("datetime", "_datetime") and name == "time":
            return TolerantTime

        # kill all bogus created_at globals (dtype etc.)
        if module == "created_at":
            return AnyShim

        return super().find_class(module, name)

def main():
    src="models/entry_now_30m_pm.pkl"
    data=open(src,"rb").read()
    bundle = FixingUnpickler(io.BytesIO(data)).load()
    print("Loaded OK:", "auc=", bundle.get("auc"), "created_at=", bundle.get("created_at"), "kind=", bundle.get("kind"))

    keep={"model","feature_names","fillna_value","pos_rate","auc","kind","dataset","created_at"}
    clean={}
    for k,v in bundle.items():
        if k in keep:
            clean[k]=v
        else:
            try:
                pickle.dumps(v, protocol=4)
                clean[k]=v
            except Exception:
                clean[k]=str(v)

    dst="models/entry_now_30m_pm_clean.pkl"
    pickle.dump(clean, open(dst,"wb"), protocol=4)
    print("Wrote", dst)

if __name__ == "__main__":
    main()
