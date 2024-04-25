import argparse
import random

random.seed(42)

# Apply the edits of a single annotator to generate the corrected sentences.
def main(args):
	m2 = open(args.m2_file).read().strip().split("\n\n")
	out = open(args.out, "w")
	# Do not apply edits with these error types
	skip = {"noop", "UNK", "Um"}
	
	for sent in m2:
		sent = sent.split("\n")
		cor_sent = sent[0].split()[1:] # Ignore "S "
		orig = " ".join(cor_sent)
		edits = sent[1:]
		offset = 0

		coders = set()
		for edit in edits:
			edit = edit.split("|||")
			if edit[1] in skip: continue # Ignore certain edits
			coder = int(edit[-1])
			coders.add(coder)

		if len(coders) > 0:
			id = random.choice(list(coders))
		else:
			id = 0

		for edit in edits:
			edit = edit.split("|||")
			if edit[1] in skip: continue # Ignore certain edits
			coder = int(edit[-1])
			# if coder != args.id: continue # Ignore other coders
			if coder != id: continue # Ignore other coders
			span = edit[0].split()[1:] # Ignore "A "
			start = int(span[0])
			end = int(span[1])
			cor = edit[2].split()
			cor_sent[start+offset:end+offset] = cor
			offset = offset-(end-start)+len(cor)
		out.write(" ".join(cor_sent) + "\t" + orig + "\n")

def parse_args():
    # Define and parse program input
	parser = argparse.ArgumentParser()
	parser.add_argument("m2_file", help="The path to an input m2 file.")
	parser.add_argument("-out", help="A path to where we save the output corrected text file.", required=True)
	# parser.add_argument("-id", help="The id of the target annotator in the m2 file.", type=int, default=0)
	return parser.parse_args()

def main_cli():
    args = parse_args()
    main(args)

if __name__ == "__main__":
    main_cli()
