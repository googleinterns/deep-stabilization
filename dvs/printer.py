import sys

class Printer(object):
    def __init__(self, *files):
        self.files = files
        
    #Redirect Printer
    def open(self):
        if not hasattr(sys, '_stdout'):
            sys._stdout = sys.stdout
        sys.stdout = self
        return self

    #Restore the Default Printer
    def close(self):
        stdout = sys._stdout
        for f in self.files:
            if f != stdout:
                f.close()
        sys.stdout = stdout

    #Overloading write() Function
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        pass

if __name__ == '__main__':
    print("Start testing")
    t = Printer(sys.stdout, open('./test.txt', 'w+')).open()
    print("In files")
    t.close()
    print("Not in files")