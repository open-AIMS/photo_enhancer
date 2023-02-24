import time

class Timer():
    def __init__(self):
        self.startTime = 0
        self.timeTaken = 0
        self.enable()

    def start(self):
        self.startTime = time.process_time()

    def stop(self):
        self.timeTaken = time.process_time() - self.startTime

    def stop_and_disp(self, label):
        self.stop()
        self.print(label)

    def print(self, label):
        if self.printEnabled:
            print(f'{label}: {self.timeTaken}')

    def disable(self):
        self.printEnabled = False

    def enable(self):
        self.printEnabled = True