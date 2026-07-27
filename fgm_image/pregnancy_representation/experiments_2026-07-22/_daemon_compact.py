import os,sys
if os.fork()>0: os._exit(0)
os.setsid()
if os.fork()>0: os._exit(0)
log=open("compact.log","w"); os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
os.dup2(os.open(os.devnull,os.O_RDONLY),0)
os.execv(sys.executable,[sys.executable,"-u","compact_summaries.py"])
