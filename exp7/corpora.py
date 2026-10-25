"""Sample corpora with gold-standard reference triples."""

from __future__ import annotations

from typing import Any, Dict


CORPORA: Dict[str, Dict[str, Any]] = {
    "Tech & Business": {
        "name": "Technology and Enterprise",
        "level": "Beginner",
        "blurb": "Concise factual statements on technology companies in straightforward active voice.",
        "purpose": (
            "Every sentence is in the active voice, consists of clear subject-verb-object syntax, "
            "and places the relationship verb directly between the two entities. This represents "
            "the baseline scenario where rule-based extraction performs with high precision."
        ),
        "observe": [
            "Observe that 'Larry Page' and 'Google' are linked by 'founded' with confidence >= 0.90.",
            "Notice how 'Google acquired YouTube' generates a high-confidence 'acquired' predicate.",
            "Compare how 'Sundar Pichai is the CEO of Google' matches the 'leads' regex rule."
        ],
        "text": (
            "Larry Page and Sergey Brin co-founded Google in California in 1998. "
            "Google acquired YouTube in 2006 and developed the Android operating system. "
            "Sundar Pichai is the CEO of Google and reports to the Alphabet board. "
            "Alphabet is headquartered in Mountain View. "
            "Microsoft, based in Redmond, competes with Google in cloud computing. "
            "Satya Nadella leads Microsoft as its chief executive. "
            "Microsoft acquired LinkedIn and partnered with OpenAI."
        ),
        "gold": [
            ("Larry Page", "founded", "Google"),
            ("Sergey Brin", "founded", "Google"),
            ("Google", "located_in", "California"),
            ("Google", "acquired", "YouTube"),
            ("Google", "created", "Android"),
            ("Sundar Pichai", "leads", "Google"),
            ("Sundar Pichai", "reports_to", "Alphabet"),
            ("Alphabet", "located_in", "Mountain View"),
            ("Microsoft", "located_in", "Redmond"),
            ("Microsoft", "competes_with", "Google"),
            ("Satya Nadella", "leads", "Microsoft"),
            ("Microsoft", "acquired", "LinkedIn"),
            ("Microsoft", "collaborates_with", "OpenAI"),
        ]
    },
    "Science & Academia": {
        "name": "Science and Academia",
        "level": "Intermediate",
        "blurb": "Biographical assertions concerning scientists, universities, and discoveries.",
        "purpose": (
            "Combines PERSON, ORG, LOCATION, and DATE entities. Demonstrates the need for entity type "
            "constraints (e.g. 'born_in' accepts a PERSON subject and LOCATION or DATE object)."
        ),
        "observe": [
            "In 'Pierre Curie, her husband', the relationship is expressed via an appositive noun phrase.",
            "Notice that Marie Curie is born in Warsaw (LOCATION) and studied at University of Paris (ORG).",
            "Observe how Albert Einstein is linked to 'theory of relativity' and collaborators."
        ],
        "text": (
            "Marie Curie was born in Warsaw and studied at the University of Paris. "
            "Curie discovered radium and polonium alongside Pierre Curie, her husband. "
            "The Curie Institute was established in Paris to continue scientific research. "
            "Albert Einstein published his theory of relativity in 1905. "
            "Einstein collaborated with Niels Bohr on quantum mechanics debates."
        ),
        "gold": [
            ("Marie Curie", "born_in", "Warsaw"),
            ("Marie Curie", "studied_at", "University of Paris"),
            ("Curie", "discovered", "radium"),
            ("Curie", "married_to", "Pierre Curie"),
            ("Curie Institute", "located_in", "Paris"),
            ("Albert Einstein", "authored", "theory of relativity"),
            ("Einstein", "collaborates_with", "Niels Bohr"),
        ]
    },
    "Indian Industry & Space": {
        "name": "Indian Industry and Space Technology",
        "level": "Intermediate",
        "blurb": "Institutional and executive statements from Indian industry and space exploration.",
        "purpose": (
            "Demonstrates extraction over domain-specific organizations and regional geographical entities."
        ),
        "observe": [
            "Look at the acquisition of Jaguar Land Rover by Tata Group.",
            "Examine how 'headquartered in Bengaluru' is detected as a located_in relationship.",
            "Observe the CEO and birthplace assertions regarding Sundar Pichai."
        ],
        "text": (
            "Ratan Tata was the chairman of the Tata Group for more than two decades. "
            "The Tata Group is headquartered in Mumbai. "
            "Tata Group acquired Jaguar Land Rover in 2008. "
            "Narayana Murthy and Nandan Nilekani founded Infosys. "
            "Infosys is headquartered in Bengaluru and competes with Tata Consultancy Services. "
            "The Indian Space Research Organisation is based in Bengaluru. "
            "Sundar Pichai was born in Chennai. "
            "Sundar Pichai is the CEO of Google."
        ),
        "gold": [
            ("Ratan Tata", "leads", "Tata Group"),
            ("Tata Group", "located_in", "Mumbai"),
            ("Tata Group", "acquired", "Jaguar Land Rover"),
            ("Narayana Murthy", "founded", "Infosys"),
            ("Nandan Nilekani", "founded", "Infosys"),
            ("Infosys", "located_in", "Bengaluru"),
            ("Infosys", "competes_with", "Tata Consultancy Services"),
            ("Indian Space Research Organisation", "located_in", "Bengaluru"),
            ("Sundar Pichai", "born_in", "Chennai"),
            ("Sundar Pichai", "leads", "Google"),
        ]
    },
    "Healthcare & Medicine": {
        "name": "Healthcare and Biomedical Research",
        "level": "Advanced",
        "blurb": "Clinical and pharmaceutical relations between institutions, therapies, and indications.",
        "purpose": (
            "Introduces specialized clinical relations (treats, causes). Demonstrates how expanding the "
            "lexicon allows rule-based extractors to adapt across technical domains."
        ),
        "observe": [
            "Observe the extraction of 'treats' between pharmacological agents and clinical conditions.",
            "Notice institutional affiliations connecting researchers to research hospitals."
        ],
        "text": (
            "Alexander Fleming discovered penicillin in London in 1928. "
            "Penicillin treats bacterial infections effectively. "
            "The World Health Organisation is based in Geneva. "
            "Pfizer partnered with BioNTech to produce vaccines. "
            "Chronic smoking causes cardiovascular disease and respiratory failure."
        ),
        "gold": [
            ("Alexander Fleming", "discovered", "penicillin"),
            ("Alexander Fleming", "located_in", "London"),
            ("Penicillin", "treats", "bacterial infections"),
            ("World Health Organisation", "located_in", "Geneva"),
            ("Pfizer", "collaborates_with", "BioNTech"),
            ("smoking", "causes", "cardiovascular disease"),
        ]
    }
}
