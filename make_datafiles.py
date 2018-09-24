'''
This code is adapted from https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
'''
import os
import struct
import subprocess
import collections
import argparse
import tensorflow as tf

from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

tokenized_articles_dir = "tokenized_articles"
finished_files_dir = "finished_files"

VOCAB_SIZE = 200000

def tokenize_stories(data_dir, filename, tokenized_articles_dir):

  """Maps a whole directory of .txt files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print "Preparing to tokenize %s to %s..." % (data_dir, tokenized_articles_dir)
  instances = os.listdir(data_dir)

  # Filter instances based upon the filename
  instances = [inst for inst in instances if filename in inst]
  print instances

  # make IO list file
  print "Making list of files to tokenize..."
  with open("mapping.txt", "w") as f:
    for inst in instances:
      f.write("%s \t %s\n" % (os.path.join(data_dir, inst), os.path.join(tokenized_articles_dir, inst)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']

  print "Tokenizing %i files in %s and saving in %s..." % (len(instances), data_dir, tokenized_articles_dir)
  subprocess.call(command)
  print "Stanford CoreNLP Tokenizer has finished."
  # os.remove("mapping.txt")

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def get_art_abs(article_file, summary_file):

  article_lines = read_text_file(article_file)
  summary_lines = read_text_file(summary_file)

  # Lowercase everything
  article_lines = [line.lower() for line in article_lines]
  summary_lines = [line.lower() for line in summary_lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  article_lines = [fix_missing_period(line) for line in article_lines]
  summary_lines = [fix_missing_period(line) for line in summary_lines]

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in summary_lines])

  return article, abstract

def write_to_bin(out_file, filename, article_tag, summary_tag, makevocab=False):

  """Reads the tokenized files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print "Making bin file for filename %s..." % filename

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      # if os.path.isfile(os.path.join(tokenized_articles_dir, filename + "." + article_tag + '.txt')):
      article_file = os.path.join(tokenized_articles_dir, filename + "." + article_tag + '.txt')
      summary_file = os.path.join(tokenized_articles_dir, filename + "." + summary_tag + '.txt')

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(article_file, summary_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print "Finished writing file %s\n" % out_file

  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file"

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Reads an article,summary from a text file and writes it in bin format.')
  parser.add_argument('--data_dir', type=str, help='The directory where raw text file resides.')
  parser.add_argument('--filename', default='sample1', type=str, help='The filename only at sample level.')
  parser.add_argument('--article_tag', default='newsarticle', type=str, help='The tag present in filename corresponding to articles.')
  parser.add_argument('--summary_tag', default='', type=str, help='The tag present in filename corresponding to the summaries.')

  args = parser.parse_args()

  data_dir = args.data_dir
  filename = args.filename
  article_tag = args.article_tag
  summary_tag = args.summary_tag

  base_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
  tokenized_articles_dir = os.path.join(base_dir, tokenized_articles_dir)
  finished_files_dir = os.path.join(base_dir, finished_files_dir)

  # Create some new directories
  if not os.path.exists(tokenized_articles_dir): os.makedirs(tokenized_articles_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on articles dir, outputting to tokenized article dir
  tokenize_stories(data_dir=args.data_dir, filename=args.filename, tokenized_articles_dir=tokenized_articles_dir)

  # Read the tokenized articles, do a little postprocessing then write to bin files
  write_to_bin(os.path.join(finished_files_dir, "test.bin"), filename=filename, article_tag=article_tag, summary_tag=summary_tag)
