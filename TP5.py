# idl_10 : Construction d'un HMM pour l'étiquetage morphosyntaxique automatique

# Exercice 0

class HMM:
    def __init__(self, initial_prob, transition_prob, emission_prob, backoff=None):
        self.tags = sorted(emission_prob.keys())
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.backoff = backoff

    def initial(self, tag):
        return self.initial_prob.get(tag, self.backoff)

    def transition(self, tag_p, tag_c):
        return self.transition_prob.get(tag_p, {}).get(tag_c, self.backoff)

    def emission(self, tag, token):
        return self.emission_prob.get(tag, {}).get(token, self.backoff)

    def __repr__(self):
        return f"HMM({self.initial_prob}, {self.transition_prob}, {self.emission_prob}, {self.backoff})"

    def __str__(self):
        return f"HMM({self.initial_prob}, {self.transition_prob}, {self.emission_prob}, {self.backoff})"



# Exercice 1
        
exo1_initial = {"DET": 1.0}

exo1_transition = {
    "ADJ": {"NOUN": 1.0},
    "CLO": {"VERB": 1.0},
    "CLS": {"VERB": 1.0},
    "DET": {"NOUN": 0.8, "ADJ": 0.2},
    "NOUN": {"CLO": 0.5, "VERB": 0.5},
    "VERB": {"DET": 1.0}
}

exo1_emission = {
    "ADJ" : {"belle": 1.0},
    "DET" : {"le": 0.6, "la": 0.4},
    "NOUN": {"belle": 0.1, "porte": 0.8, "voile": 0.1},
    "VERB": {"porte": 0.6, "voile": 0.4},
    "CLO" : {"le": 1.0}
}

exo1_hmm = HMM(exo1_initial, exo1_transition, exo1_emission, 0)

assert exo1_hmm.initial("DET") == 1, exo1_hmm.initial("DET")
assert exo1_hmm.initial("ADJ") == 0, exo1_hmm.initial("ADJ")

assert exo1_hmm.transition("DET", "ADJ") == 0.2, exo1_hmm.transition("DET", "ADJ")
assert exo1_hmm.transition("DET", "NOUN") == 0.8, exo1_hmm.transition("DET", "NOUN")
assert exo1_hmm.transition("DET", "VERB") == 0.0, exo1_hmm.transition("DET", "VERB")

assert exo1_hmm.emission("DET", "le") == 0.6, exo1_hmm.emission("DET", "le")
assert exo1_hmm.emission("DET", "la") == 0.4, exo1_hmm.emission("DET", "la")
assert exo1_hmm.emission("DET", "les") == 0.0, exo1_hmm.emission("DET", "les")

# a. Calculez, pour chaque étiquette E, la probabilité de "la" sachant E.


def initials(hmm, token):
    res = {}
    for tag in hmm.tags:
        res[tag] = {
            "probabilité": hmm.initial(tag) * hmm.emission(tag, token),
            "depuis": None
        }
    return res

assert initials(exo1_hmm, "la") == {
    'ADJ': {'probabilité': 0.0, 'depuis': None},
    'CLO': {'probabilité': 0.0, 'depuis': None},
    'DET': {'probabilité': 0.4, 'depuis': None},
    'NOUN': {'probabilité': 0.0, 'depuis': None},
    'VERB': {'probabilité': 0.0, 'depuis': None}
}, initials(exo1_hmm, "la")

# b. Calculez la meilleure transition vers l'étiquette ADJ

def best_transition_to(hmm, probas_preced, etiquette, token):
    res = {"probabilité": -float('inf'), "depuis": None}
    for tag in hmm.tags:
        proba = probas_preced[tag]["probabilité"] * hmm.transition(tag, etiquette)
        if proba > res["probabilité"]:
            res["probabilité"] = proba
            res["depuis"] = tag
    res["probabilité"] *= hmm.emission(etiquette, token)
    return res

avant = initials(exo1_hmm, "la")
assert (
    best_transition_to(exo1_hmm, avant, "ADJ", "belle") == {'probabilité': 0.08000000000000002, 'depuis': 'DET'}
), best_transition_to(exo1_hmm, avant, "ADJ", "belle")

# c. Calculez la meilleure transition vers toutes les étiquettes


def best_transitions(hmm, probas_preced, token):
    res = {}
    for tag in hmm.tags:
        res[tag] = best_transition_to(hmm, probas_preced, tag, token)
    return res

    
avant = initials(exo1_hmm, "la")
assert best_transitions(exo1_hmm, avant, "belle") == {
    'ADJ': {'probabilité': 0.08000000000000002, 'depuis': 'DET'},
    'CLO': {'probabilité': 0.0, 'depuis': 'ADJ'},
    'DET': {'probabilité': 0.0, 'depuis': 'ADJ'},
    'NOUN': {'probabilité': 0.03200000000000001, 'depuis': 'DET'},
    'VERB': {'probabilité': 0.0, 'depuis': 'ADJ'}
}, best_transitions(exo1_hmm, avant, "belle")

# d. Construire la matrice de Viterbi

def viterbi_matrix(hmm, words):
    
    matrix = []
    for i, word in enumerate(words):
        if i == 0:
            matrix.append(initials(hmm, word))
        else:
            matrix.append(best_transitions(hmm, matrix[i-1], word))
    return matrix

matrix = viterbi_matrix(exo1_hmm, ["la", "belle", "porte", "le", "voile"])

def non_zeroes(d):
    return {tag: value for tag, value in d.items() if value["probabilité"] != 0}

assert non_zeroes(matrix[0]) == {'DET': {'probabilité': 0.4, 'depuis': None}}, non_zeroes(matrix[0])
assert non_zeroes(matrix[1]) == {'ADJ': {'probabilité': 0.08000000000000002, 'depuis': 'DET'}, 'NOUN': {'probabilité': 0.03200000000000001, 'depuis': 'DET'}}, non_zeroes(matrix[1])
assert non_zeroes(matrix[2]) == {'NOUN': {'probabilité': 0.06400000000000002, 'depuis': 'ADJ'}, 'VERB': {'probabilité': 0.009600000000000003, 'depuis': 'NOUN'}}, non_zeroes(matrix[2])
assert non_zeroes(matrix[3]) == {'CLO': {'probabilité': 0.03200000000000001, 'depuis': 'NOUN'}, 'DET': {'probabilité': 0.005760000000000001, 'depuis': 'VERB'}}, non_zeroes(matrix[3])
assert non_zeroes(matrix[4]) == {'NOUN': {'probabilité': 0.00046080000000000014, 'depuis': 'DET'}, 'VERB': {'probabilité': 0.012800000000000004, 'depuis': 'CLO'}}, non_zeroes(matrix[4])

# e. Reconstruire la séquence

def viterbi(hmm, sentence):
    matrix = viterbi_matrix(hmm, sentence)
    last = matrix[-1]
    best = max(last.values(), key=lambda x: x["probabilité"])
    tags = [max(last, key=lambda x: last[x]["probabilité"])]
    for i in range(len(sentence)-1, 0, -1):
        tags.append(matrix[i][tags[-1]]["depuis"])
    return list(reversed(tags)), best["probabilité"]

tags, prob = viterbi(exo1_hmm, ["la", "belle", "porte", "le", "voile"])
assert tags == ['DET', 'ADJ', 'NOUN', 'CLO', 'VERB'], tags
assert prob == 0.012800000000000004, prob

# Exercice 2 : créer un HMM depuis un jeu d'entraînement

def lire(path):
    with open(path, "r") as f:
        return [[token.rsplit("/", 1) for token in line.strip().split()] for line in f] 
    # évite de sauter les '/' en tant que token, ne segmente qu'à la dernière occurence de '/'

sent = lire("test_lire.txt")
assert sent == [[['Que', 'SCONJ'], ['la', 'DET'], ['lumière', 'NOUN'], ['soit', 'VERB'], ['!', 'PUNCT']]], sent

def depuis_corpus(corpus):
    initials = {}
    emissions = {}
    transitions = {}

    for sentence in corpus:
        initials[sentence[0][1]] = initials.get(sentence[0][1], 0) + 1
        
        for i in range(len(sentence)):
            tag, word = sentence[i][1], sentence[i][0]
            emissions.setdefault(tag, {}).setdefault(word, 0)
            emissions[tag][word] += 1
            
            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                transitions.setdefault(tag, {}).setdefault(next_tag, 0)
                transitions[tag][next_tag] += 1

    total_initials = sum(initials.values())
    for tag in initials:
        initials[tag] /= total_initials

    for tag in emissions:
        total_emissions = sum(emissions[tag].values())
        for word in emissions[tag]:
            emissions[tag][word] /= total_emissions

    for tag in transitions:
        total_transitions = sum(transitions[tag].values())
        for next_tag in transitions[tag]:
            transitions[tag][next_tag] /= total_transitions

    return HMM(initials, transitions, emissions, 0)

train = lire("sequoia/fr_sequoia-ud-train.line.txt")
my_hmm = depuis_corpus(train)
sentence = ["Le", "professeur", "Gaston", "parle"]
path, probs = viterbi(my_hmm, sentence)

assert path == ['DET', 'NOUN', 'PROPN', 'VERB'], path
assert abs(probs - 7.697081342015525e-17) < 1e-20

print("proba =", probs)
# La proba peut différer légèrement, mais devrait afficher :
# proba = 7.697081342015525e-17

dev = lire("sequoia/fr_sequoia-ud-dev.line.txt")
total = 0
correct = 0
for sentence in dev:
    tokens = [token for token, tag in sentence]
    gold = [tag for token, tag in sentence]
    total += len(sentence)
    guess, prob = viterbi(my_hmm, tokens)
    correct += sum(guess[i] == gold[i] for i in range(len(gold)))
print("L'accuracy du HMM sur le corpus de développement est de :", 100*correct / total, "%")

# Le one-count smoothing

def depuis_corpus(corpus):
    initials = {}
    emissions = {}
    transitions = {}

    all_words = set()
    all_tags = set()
    for sentence in corpus:
        for token, tag in sentence:
            all_words.add(token)
            all_tags.add(tag)

    n = sum(len(sentence) for sentence in corpus)
    V = len(all_words) + len(all_tags)

    for sentence in corpus:
        initials[sentence[0][1]] = initials.get(sentence[0][1], 0) + 1
        
        for i in range(len(sentence)):
            tag, word = sentence[i][1], sentence[i][0]
            emissions.setdefault(tag, {}).setdefault(word, 0)
            emissions[tag][word] += 1
            
            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                transitions.setdefault(tag, {}).setdefault(next_tag, 0)
                transitions[tag][next_tag] += 1

    total_initials = sum(initials.values())
    total_emissions = sum(sum(emissions[tag].values()) for tag in emissions)
    total_transitions = sum(sum(transitions[tag].values()) for tag in transitions)

    backoff_emission = 1 / (n + V)

    for tag in initials:
        initials[tag] /= total_initials

    for tag in emissions:
        for word in emissions[tag]:
            emissions[tag][word] /= sum(emissions[tag].values())

    for tag in transitions:
        for next_tag in transitions[tag]:
            transitions[tag][next_tag] /= sum(transitions[tag].values())

    return HMM(initials, transitions, emissions, backoff_emission)


train = lire("sequoia/fr_sequoia-ud-train.line.txt")
dev = lire("sequoia/fr_sequoia-ud-dev.line.txt")
my_hmm = depuis_corpus(train)

total = 0
correct = 0
for sentence in dev:
    tokens = [token for token, tag in sentence]
    gold = [tag for token, tag in sentence]
    total += len(sentence)
    guess, prob = viterbi(my_hmm, tokens)
    correct += sum(guess[i] == gold[i] for i in range(len(gold)))
print("L'accuracy du HMM sur le corpus de développement est de :", 100*correct / total, "%")