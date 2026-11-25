import sys, traceback
try:
    import train_clip as tc
    tc.CFG.epochs = 5
    tc.train()
except Exception as e:
    traceback.print_exc()
    with open("train_crash.log", "w") as f:
        f.write(traceback.format_exc())
