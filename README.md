Επεξεργασία Φυσικής Γλώσσας 2025

Αυτό το αποθετήριο περιέχει την υλοποίηση για την εργασία του μαθήματος "Επεξεργασία Φυσικής Γλώσσας". Ο στόχος του έργου είναι ο μετασχηματισμός μη δομημένων κειμένων σε σαφείς, ορθές και καλά δομημένες εκδοχές, χρησιμοποιώντας τεχνικές NLP. Η ανάλυση των αποτελεσμάτων γίνεται μέσω υπολογισμού σημασιολογικής ομοιότητας (cosine similarity) και οπτικοποίησης των διανυσματικών αναπαραστάσεων (word embeddings) με PCA.


Setup and Installation

Για την εκτέλεση, ακολουθήστε τα παρακάτω βήματα.

Prerequisites
* [Conda](https://www.anaconda.com/products/distribution)
* Python >= 3.11
* [Poetry](https://python-poetry.org/)

Installation Steps:

1.  Κλωνοποιήστε το αποθετήριο:
    git clone https://github.com/DIONISISPX/nlp2025.git
    cd nlp

2.  Δημιουργήστε και ενεργοποιήστε το περιβάλλον Conda:
    conda create -name nlp python=3.11
    conda activate nlp

3.  Εγκαταστήστε τις εξαρτήσεις του έργου με το Poetry:
    poetry install

How to Run

Μπορείτε να εκτελέσετε τα πειράματα χρησιμοποιώντας τις παρακάτω εντολές από τον κύριο φάκελο του έργου ή με το περιβάλλον που επιθυμείτε (πχ VS Code).

1.  Εκτέλεση της ανακατασκευής με custom κανόνες (Παραδοτέο 1Α):
    poetry run python source/partA.py

2.  Εκτέλεση της ανακατασκευής με τα pipeline models (Παραδοτέο 1Β):
    poetry run python source/partB.py

3.  Εκτέλεση της υπολογιστικής ανάλυσης και οπτικοποίησης (Παραδοτέο 2):
    poetry run python source/analysis.py

Η αναλυτική συζήτηση των αποτελεσμάτων, των προκλήσεων και των ευρημάτων περιλαμβάνεται στο αρχείο `Structured Report.pdf`.
