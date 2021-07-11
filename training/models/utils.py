import os


# Helper class for storing training outputs
class Logger():
  def __init__(self, base_dir):
    self.base_dir = base_dir

  def log(self, text, rel_path, mode='a'):
    filepath = os.path.join(
      self.base_dir, rel_path
    )

    f = open(filepath, mode)
    f.write(text + '\n')
    f.close()

    print(text)

  def reprint(self, rel_path):
    filepath = os.path.join(
      self.base_dir, rel_path
    )

    f = open(filepath, 'r')
    text = f.read()
    f.close()

    print(text[:-1])  # Ignore last `\n`
