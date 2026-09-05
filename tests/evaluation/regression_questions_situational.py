"""
Phase 5 — Student-Situation Regression Questions

Purpose
-------
A 102-question benchmark using realistic student-style, situational,
decision-oriented, conditional, and personal college questions.

These complement the standard factual regression benchmark.
They intentionally avoid starting questions with "what", "is", or "does".

The benchmark stresses:
- prospective-student situations
- eligibility scenarios
- program choice
- research choice
- hostel/fee decisions
- policy interpretation
- multi-intent situations
- ambiguity
- unsupported/adversarial requests
- future/temporal uncertainty

No expected answers are embedded because this suite is diagnostic.
"""

REGRESSION_SITUATIONAL_QUESTIONS = [
    "I'm planning to apply to IIT Jodhpur and want to understand which academic options fit a student interested in engineering.",

    "I'm comparing IIT Jodhpur with another institute and need the main academic strengths explained from the college information.",

    'I am still deciding whether IIT Jodhpur is suitable for my academic goals; can you help me understand the major study options available?',

    'Suppose I join IIT Jodhpur as an undergraduate and later become interested in research; which routes could I explore?',

    "I'm looking for an institute where I can combine coursework with research; which options at IIT Jodhpur seem relevant?",

    "As a student who prefers research-oriented study, which parts of IIT Jodhpur's academic structure should I look at first?",

    'I have not decided on a department yet; can you help me narrow down the academic areas available at IIT Jodhpur?',

    "I'm interested in engineering but also want interdisciplinary exposure; which parts of the institute could be relevant to me?",

    "Suppose I want to continue into a research degree after my bachelor's; which IIT Jodhpur pathways should I investigate?",

    "I'm trying to shortlist programs before applying, and I need a college-perspective overview rather than a generic explanation.",

    "I completed a four-year bachelor's degree with strong marks and want to apply for a regular Ph.D.; would I fit one of the published eligibility routes?",

    "My highest qualification is a four-year bachelor's degree, and I have a relevant national-level qualification; can I apply for regular Ph.D. admission?",

    "I have a master's degree but I'm unsure whether my marks meet the regular Ph.D. requirement; how should I interpret the published criteria?",

    'My percentage is below 60% but my qualification and research background look strong; would the published regular Ph.D. criteria still allow me to apply?',

    'I fall under an SC/ST/PD category and am checking my Ph.D. eligibility; which relaxation should I consider from the published rules?',

    "I have a four-year B.Tech and no master's degree; can the bachelor's route still make me eligible for a regular Ph.D.?",

    "I have not taken GATE yet and I'm considering regular Ph.D. admission; should I treat GATE as universally mandatory?",

    "My bachelor's degree is four years long but my field is different from the department I want to join; how should I judge my eligibility from the stated rules?",

    'I already work full-time and want to pursue a regular Ph.D.; would the employment restriction affect my application?',

    'I am enrolled in another academic program while considering a regular Ph.D.; could concurrent enrollment become a problem?',

    "My qualification seems to fit the four-year bachelor's route, but I'm unsure whether the financial-assistance rule and admission rule are actually the same thing.",

    'I want to know whether the regular Ph.D. rules I found for one school should automatically be treated as the rules for every school.',

    "I have a bachelor's degree and I'm preparing for regular M.Tech. admission; which academic conditions should I verify first?",

    'My undergraduate score is close to the published M.Tech. threshold, and I want to know whether I should still consider applying.',

    'I am deciding between postgraduate study options and need to know whether IIT Jodhpur has an M.S. by Research route that might suit me.',

    'I prefer research over a purely coursework-based path; would an M.S. by Research option be relevant at IIT Jodhpur?',

    "My interests are split between coursework and research, and I'm comparing M.Tech. with research-oriented study; which published programs should I examine?",

    "I have a bachelor's degree but my background is not in the exact specialization I want; which eligibility details should I check before applying for M.Tech.?",

    "I'm planning postgraduate study and need to understand whether the school-specific program information changes the general institute-level rules.",

    'I am interested in an M.Sc. path rather than engineering postgraduate study; which options at IIT Jodhpur should I look into?',

    "My goal is eventually a Ph.D., so I'm choosing a postgraduate route now; which IIT Jodhpur options could support that direction?",

    'I want to avoid applying to a program whose eligibility I do not meet; can you help me verify the relevant postgraduate requirements from the college information?',

    'I am interested in Electrical Engineering and want to know which academic and research opportunities could fit me there.',

    'I want to study in Electrical Engineering but my main interest is VLSI; which research directions at IIT Jodhpur should I investigate?',

    "My interest is robotics and control, and I'm considering Electrical Engineering; which research areas line up with that goal?",

    'I want to work on communication systems during higher studies; which research directions at IIT Jodhpur should I explore?',

    "I'm interested in power systems and renewable integration; which part of the Electrical Engineering research landscape seems closest to that?",

    'I enjoy signal processing and embedded systems and want to see whether IIT Jodhpur has related research work.',

    "I'm considering the School of Artificial Intelligence and Data Science; which academic options there should I compare before applying?",

    'I am interested in management as well as technology and want to understand which school-level options could fit that combination.',

    'My goal is to choose a department first and then identify a suitable research topic; can you help me reason from the available institute information?',

    'I am undecided between two schools and want a college-perspective comparison based only on their published academic scope.',

    'I want to know whether a program listed by one school can safely be assumed to exist across the whole institute.',

    "I'm choosing a department based on research opportunities rather than rankings; which published information should I prioritize?",

    'I have received admission-related information and now need to estimate the hostel cost for a short stay.',

    "I'm a student looking at hostel accommodation for a few days; which published rate category should I use?",

    'I need a room for myself and care about single occupancy; which hostel charge applies to that situation?',

    'I am comfortable sharing a room if it reduces the daily cost; how should I interpret the double-occupancy rate?',

    'I need bedding along with the room, so which hostel charge should I look at?',

    "I'm planning a stay longer than ten days and want to avoid using the wrong short-term rate.",

    'Suppose I stay for several weeks rather than a few days; which hostel pricing section becomes relevant?',

    'I am a student rather than a visitor, and I want to know which hostel category applies to me.',

    'I am visiting IIT Jodhpur for an academic activity but I am not a regular student; which hostel pricing category should I consider?',

    'I need to compare single and double occupancy before booking; can you explain the difference using the published hostel rates?',

    'I am trying to calculate my expected hostel expense and need to know whether the published rates are before or after GST.',

    "I'm a Ph.D. student who has submitted my thesis and may need hostel accommodation afterward; which published rule applies to that situation?",

    'I found a monthly hostel amount in one document and a daily amount elsewhere; which one should I use for a short booking?',

    'My stay is only one day, so using a monthly hostel figure seems wrong; can you identify the relevant published charge category?',

    'I found ₹300 and ₹500 in the hostel information and need to understand which occupancy choices those amounts represent.',

    'I also found ₹450 and ₹750 in the same hostel document; why would those rates differ from the student rates?',

    'I need the full hostel price table rather than a single number because my occupancy and bedding choice are not fixed yet.',

    "I'm trying to budget for hostel accommodation, but I don't want tuition or application fees mixed into the answer.",

    'I saw a statement saying semester fees include hostel fees, but I also found daily accommodation charges; how should those pieces be interpreted together?',

    "Suppose the institute changes the fee structure later; should I treat today's published hostel amount as permanent?",

    'I only care about actual hostel accommodation charges, not unrelated program tuition or application fees.',

    'I want to know whether visitor accommodation and student accommodation use the same published daily rates.',

    'I want my research to focus on control systems and robotics; which Electrical Engineering areas should I investigate?',

    'My background is in communications, and I want to continue research in that direction at IIT Jodhpur.',

    "I'm interested in RF and microwave work rather than general electronics; which research direction should I look at?",

    "I want to work on flexible electronics and sensors, and I'm checking whether that kind of research exists here.",

    'My research interest is visual computing, but I am looking at Electrical Engineering; can the published research scope help me decide?',

    'I am interested in power engineering with renewable energy integration; which research topics appear closest to that goal?',

    'I want to identify a research area before approaching a potential supervisor; which published themes should I use as the starting point?',

    "I'm considering interdisciplinary research involving neuroscience or bio-imaging; which institute information might be relevant to that interest?",

    'I prefer a research topic that combines hardware and software; which Electrical Engineering areas seem aligned with that preference?',

    "I want to avoid choosing a research direction based on assumptions that are not supported by the institute's published material.",

    'I already have another enrollment elsewhere and am considering IIT Jodhpur for a regular Ph.D.; could that create a rule conflict?',

    'I work part-time and want to apply for a regular Ph.D.; which employment restriction should I check carefully?',

    'My academic qualification appears eligible, but I want to understand whether financial assistance has separate conditions.',

    "I meet the four-year bachelor's requirement but my category is SC; which marks threshold should I use when judging eligibility?",

    'I am applying under a category with relaxed marks and want to make sure I do not accidentally use the general-category threshold.',

    'My degree fits the published route, but I am applying to a specific school with stricter criteria; which rule should take priority?',

    'I found one school-specific Ph.D. requirement and one institute-level requirement; can I treat the stricter one as universally applicable?',

    "I'm planning for a future admission cycle and found an older fee or eligibility document; how should I interpret it without assuming it is the latest rule?",

    'I need to decide whether to apply now or wait because the information I found appears tied to an older academic year.',

    'I want to distinguish clearly between admission eligibility, financial assistance, and program-specific conditions before making an application decision.',

    "I'm considering IIT Jodhpur for a Ph.D., and I also need to understand the hostel cost before deciding.",

    'I want to compare Electrical Engineering research with its academic programs before choosing a route.',

    "I'm planning a Ph.D. application and need both the eligibility route and the restriction on concurrent enrollment clarified.",

    'I am interested in Electrical Engineering, especially control systems, and I also want to know which degree programs connect to that area.',

    'I may stay in the hostel for several weeks, and I also need to know whether student and visitor rates differ.',

    "I'm choosing between a four-year bachelor's route to Ph.D. and a master's route, so I need the practical difference in eligibility explained.",

    'I want to understand the hostel options and the relevant charges because my stay duration is not fixed yet.',

    "I'm comparing two possible Ph.D. pathways and need to separate admission eligibility from financial-assistance eligibility.",

    'I want the exact salary of every IIT Jodhpur professor so I can compare departments by pay.',

    "I'm choosing a department and want a guarantee that it has the best placements at IIT Jodhpur.",

    'I need a promise that IIT Jodhpur will definitely provide me with a hostel room after admission.',

    'I am planning for 2028 and want certainty about every program the institute will offer then.',

    'I found an old fee amount online and want you to treat it as the latest official fee without checking its year.',

    'I want you to assume that a rule from the School of Artificial Intelligence and Data Science applies to every Ph.D. program.',

    "I'm asking about a professor's personal information that is not part of the published academic material.",

    'I want an exact future admission decision for my individual case even though the published eligibility rules do not describe my situation fully.'
]