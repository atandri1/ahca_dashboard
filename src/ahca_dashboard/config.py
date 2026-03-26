CMS_REGIONS = {
    "reg1": {"region": 1, "states": ["CT", "ME", "MA", "NH", "RI", "VT"]},
    "reg2": {"region": 2, "states": ["NJ", "NY", "PR", "VI"]},
    "reg3": {"region": 3, "states": ["DE", "DC", "MD", "PA", "VA", "WV"]},
    "reg4": {"region": 4, "states": ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN"]},
    "reg5": {"region": 5, "states": ["IL", "IN", "MI", "MN", "OH", "WI"]},  # covers 5a and 5b
    "reg6": {"region": 6, "states": ["AR", "LA", "NM", "OK", "TX"]},
    "reg7": {"region": 7, "states": ["IA", "KS", "MO", "NE"]},
    "reg8": {"region": 8, "states": ["CO", "MT", "ND", "SD", "UT", "WY"]},
    "reg9": {"region": 9, "states": ["AZ", "CA", "HI", "NV", "AS", "GU", "MP"]},
    "reg10": {"region": 10, "states": ["AK", "ID", "OR", "WA"]},
}

# Top 10 deficiency tags (J/K/L severity) referenced in the notebook.
TARGET_TAGS = [580, 600, 610, 678, 684, 686, 689, 760, 835, 880]

TAG_DESCRIPTIONS = {
    580: "F-0580: Notify of Changes (Physician/Family)",
    600: "F-0600: Free from Abuse and Neglect",
    610: "F-0610: Investigate/Prevent/Correct Abuse",
    678: "F-0678: Cardio-Pulmonary Resuscitation (CPR)",
    684: "F-0684: Quality of Care",
    686: "F-0686: Treatment/Services to Prevent/Heal Pressure Ulcers",
    689: "F-0689: Free of Accident Hazards/Supervision/Devices",
    760: "F-0760: Medication Regimen Free from Unnecessary Drugs",
    835: "F-0835: Administration",
    880: "F-0880: Infection Prevention & Control",
}

SEVERITY_LEVELS = {
    "J": "Isolated - Immediate Jeopardy",
    "K": "Pattern - Immediate Jeopardy",
    "L": "Widespread - Immediate Jeopardy",
}

DEFAULT_MIN_YEAR = 2022

