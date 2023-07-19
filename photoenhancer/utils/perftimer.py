import time

class Timer():
    def __init__(self):
        self.startTime = 0
        self.timeTaken = 0
        self.enable()
        self.labels={}

    def rest(self):
        self.labels = {}

    def start(self):
        self.startTime = time.process_time()

    def stop(self):
        self.timeTaken = time.process_time() - self.startTime

    def stop_and_disp(self, label):
        self.stop()
        self.print(label)
        label_time = self.labels.get(label)
        if label_time is None:
            label_time = 0
        label_time += self.timeTaken
        self.labels[label] = label_time

    def print_labels(self):
        print ("Times for batch\n")
        for label, time in self.labels.items():
            print(f'{label}: {time}\n')

    def print(self, label):
        if self.printEnabled:
            print(f'{label}: {self.timeTaken}')

    def disable(self):
        self.printEnabled = False

    def enable(self):
        self.printEnabled = True