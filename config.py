# ─────────────────────────────────────────────────────────────────────────────
# RecruitAI — Configuration & Constants
# ─────────────────────────────────────────────────────────────────────────────
"""
Central configuration for the RecruitAI ranking pipeline.
All weights, thresholds, skill mappings, and company classifications live here
so they can be tuned without touching pipeline logic.
"""

# ── Scoring Dimension Weights (must sum to 1.0) ──────────────────────────────
WEIGHTS = {
    "title_fit":       0.25,
    "skills_match":    0.20,
    "career_quality":  0.20,
    "experience_band": 0.10,
    "education_fit":   0.05,
    "location_fit":    0.05,
    "behavioral":      0.15,
}

# ── Final Composite Blend ─────────────────────────────────────────────────────
RULE_WEIGHT    = 0.55   # Weight for rule-based score in final blend
SEMANTIC_WEIGHT = 0.45  # Weight for semantic similarity in final blend

# ── Stage Thresholds ──────────────────────────────────────────────────────────
STAGE2_TOP_N   = 2000   # How many candidates pass to semantic scoring
STAGE3_TOP_N   = 200    # How many candidates pass to final ranking
FINAL_TOP_N    = 100    # Final output size

# ── Experience Band ───────────────────────────────────────────────────────────
IDEAL_EXP_MIN = 5.0
IDEAL_EXP_MAX = 9.0
IDEAL_EXP_CENTER = 7.0  # Sweet spot per JD ("6-8 years")
EXP_HARD_MIN = 2.0      # Below this, very unlikely fit
EXP_HARD_MAX = 18.0     # Above this, likely too senior / different trajectory

# ── Title Classification ──────────────────────────────────────────────────────
# Scores for current/past titles indicating fit for "Senior AI Engineer"
TITLE_SCORES = {
    # Exact / near-exact fit (90-100)
    "ai engineer":                   98,
    "senior ai engineer":            100,
    "machine learning engineer":     97,
    "senior machine learning engineer": 100,
    "ml engineer":                   97,
    "senior ml engineer":            100,
    "deep learning engineer":        95,
    "nlp engineer":                  95,
    "applied scientist":             93,
    "applied ml scientist":          95,
    "research engineer":             88,
    "ai/ml engineer":                97,
    "ml platform engineer":          90,
    "mlops engineer":                85,

    # Strong adjacent (70-89)
    "data scientist":                82,
    "senior data scientist":         85,
    "lead data scientist":           85,
    "principal data scientist":      85,
    "software engineer":             70,
    "senior software engineer":      75,
    "backend engineer":              68,
    "senior backend engineer":       72,
    "platform engineer":             65,
    "data engineer":                 65,
    "senior data engineer":          68,
    "full stack engineer":           55,

    # Partially related (30-59)
    "data analyst":                  40,
    "analytics engineer":            45,
    "devops engineer":               35,
    "cloud engineer":                35,
    "site reliability engineer":     30,
    "product manager":               25,
    "technical product manager":     35,
    "engineering manager":           40,
    "tech lead":                     50,
    "cto":                           45,
    "junior ml engineer":            70,
    "junior data scientist":         60,
    "junior software engineer":      45,

    # Unrelated (0-20)
    "marketing manager":             5,
    "sales executive":               3,
    "hr manager":                    3,
    "accountant":                    2,
    "operations manager":            5,
    "business analyst":              20,
    "project manager":               15,
    "content writer":                5,
    "graphic designer":              5,
    "mechanical engineer":           8,
    "civil engineer":                5,
    "customer support":              3,
    "electrical engineer":           10,
    "financial analyst":             5,
    "teacher":                       2,
    "consultant":                    15,
    "management consultant":         15,
}

# ── Skills Classification ─────────────────────────────────────────────────────
# Must-have skills from JD (highest weight)
MUST_HAVE_SKILLS = {
    "sentence-transformers", "sentence transformers", "embeddings",
    "vector databases", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "opensearch", "elasticsearch",
    "python", "ranking systems", "retrieval systems",
    "ndcg", "mrr", "map", "evaluation frameworks",
    "information retrieval", "search systems", "hybrid search",
}

# Core AI/ML skills (high weight)
CORE_AI_SKILLS = {
    "pytorch", "tensorflow", "nlp", "natural language processing",
    "transformers", "hugging face", "huggingface", "bert", "gpt",
    "deep learning", "machine learning", "neural networks",
    "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "recommendation systems", "recommender systems",
    "feature engineering", "model training", "model deployment",
    "mlflow", "wandb", "weights & biases", "weights and biases",
    "bentoml", "mlops", "model serving",
    "rag", "retrieval augmented generation", "langchain",
    "llm", "large language models", "fine-tuning",
    "fine-tuning llms", "lora", "qlora", "peft",
    "prompt engineering", "openai", "anthropic",
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "gans",
    "statistical modeling", "bayesian",
    "data science", "a/b testing",
    "spark", "pyspark", "hadoop",
}

# Nice-to-have technical skills (moderate weight)
NICE_TO_HAVE_SKILLS = {
    "docker", "kubernetes", "aws", "gcp", "azure",
    "sql", "postgresql", "mongodb", "redis",
    "kafka", "airflow", "apache beam", "apache flink",
    "databricks", "snowflake", "dbt",
    "git", "ci/cd", "linux",
    "java", "scala", "go", "rust", "c++",
    "flask", "fastapi", "django",
    "graphql", "rest api", "grpc",
    "numpy", "pandas", "scipy", "matplotlib",
    "jupyter", "notebooks",
    "ray", "dask",
}

# NLP/IR specific skills (bonus for this role)
NLP_IR_SKILLS = {
    "nlp", "natural language processing", "information retrieval",
    "text mining", "text classification", "sentiment analysis",
    "named entity recognition", "ner", "tokenization",
    "word embeddings", "word2vec", "glove", "fasttext",
    "bert", "gpt", "transformers", "hugging face",
    "search", "ranking", "bm25", "tf-idf", "tfidf",
    "semantic search", "vector search",
    "elasticsearch", "solr", "lucene",
}

# Anti-signal skills (skills that suggest non-tech background)
ANTI_SIGNAL_SKILLS = {
    "photoshop", "illustrator", "indesign", "canva",
    "seo", "content writing", "copywriting",
    "excel", "powerpoint", "word",
    "accounting", "tally", "quickbooks",
    "six sigma", "lean", "pmp",
    "sap", "salesforce", "hubspot",
    "autocad", "solidworks", "creo", "catia", "ansys",
    "marketing", "social media marketing", "google ads",
    "recruitment", "talent acquisition",
}

# ── Company Classification ────────────────────────────────────────────────────
# Known consulting/services companies (career-long = penalty per JD)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree",
    "mphasis",
    "ltimindtree",
    "l&t infotech",
    "persistent systems",
    "hexaware",
    "zensar",
    "cyient",
    "niit technologies",
    "birlasoft",
    "coforge",
}

# Known product / tech companies (positive signal)
PRODUCT_COMPANIES = {
    "google", "alphabet",
    "microsoft",
    "amazon", "aws",
    "meta", "facebook",
    "apple",
    "netflix",
    "uber",
    "airbnb",
    "stripe",
    "databricks",
    "snowflake",
    "palantir",
    "nvidia",
    "openai",
    "anthropic",
    "hugging face",
    "cohere",
    "anyscale",
    "weights & biases",
    "deepmind",
    "flipkart",
    "swiggy",
    "zomato",
    "razorpay",
    "cred",
    "meesho",
    "phonepe",
    "paytm",
    "ola",
    "byju's",
    "freshworks",
    "zoho",
    "postman",
    "browserstack",
    "hasura",
    "chargebee",
    "clevertap",
    "druva",
    "icertis",
    "innovaccer",
    "moengage",
    "yellow.ai",
}

# Fictional companies in the dataset
FICTIONAL_COMPANIES = {
    "dunder mifflin", "globex inc", "globex", "initech", "acme corp",
    "stark industries", "wayne enterprises", "umbrella corp",
    "weyland-yutani", "soylent corp", "cyberdyne", "oscorp",
    "massive dynamic", "hooli", "pied piper", "raviga",
}

# ── Location Classification ───────────────────────────────────────────────────
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi ncr", "delhi", "new delhi",
    "gurgaon", "gurugram",
}

GOOD_LOCATIONS_INDIA = {
    "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai",
    "kolkata", "ahmedabad", "jaipur", "chandigarh", "lucknow",
    "kochi", "thiruvananthapuram", "indore", "nagpur",
    "gurgaon", "gurugram", "haryana", "noida", "greater noida",
    "ghaziabad", "faridabad",
}

INDIA_COUNTRY_NAMES = {"india", "in"}

# ── Behavioral Signal Weights ────────────────────────────────────────────────
BEHAVIORAL_WEIGHTS = {
    "recruiter_response_rate":      0.20,
    "last_active_recency":          0.15,
    "open_to_work":                 0.10,
    "interview_completion_rate":    0.10,
    "notice_period":                0.10,
    "profile_completeness":         0.08,
    "github_activity":              0.07,
    "avg_response_time":            0.05,
    "willing_to_relocate":          0.03,
    "verification_trust":           0.05,
    "saved_by_recruiters":          0.04,
    "search_appearances":           0.03,
}

# ── Honeypot Detection Thresholds ─────────────────────────────────────────────
HONEYPOT_EXPERT_SKILL_THRESHOLD = 5    # Expert in 5+ skills with minimal duration
HONEYPOT_SKILL_DURATION_MIN = 6        # Months — if "expert" with <6mo, suspicious
HONEYPOT_EXP_MISMATCH_RATIO = 1.8     # claimed_yrs / career_sum ratio threshold
HONEYPOT_MAX_ZERO_DURATION_EXPERTS = 3 # Expert skills with 0 duration

# ── Career Description Keywords ───────────────────────────────────────────────
# Keywords in career descriptions that indicate real ML/AI production work
ML_PRODUCTION_KEYWORDS = {
    "model", "training", "inference", "embeddings", "vector",
    "ranking", "recommendation", "retrieval", "search",
    "nlp", "natural language", "text classification",
    "deep learning", "neural network", "transformer",
    "pytorch", "tensorflow", "scikit-learn",
    "feature engineering", "pipeline", "ml pipeline",
    "a/b test", "experiment", "evaluation",
    "deployed", "production", "serving", "endpoint",
    "fine-tun", "fine tun", "bert", "gpt", "llm",
    "data science", "machine learning", "artificial intelligence",
}

# Keywords suggesting non-ML work
NON_ML_KEYWORDS = {
    "accounting", "financial reporting", "tax", "audit",
    "marketing campaign", "brand identity", "seo strategy",
    "customer support", "ticket", "escalation",
    "mechanical engineering", "cad", "solidworks", "ansys",
    "civil engineering", "construction", "structural",
    "sales", "revenue target", "quota",
    "hr", "recruitment", "talent acquisition", "payroll",
    "supply chain", "warehouse", "logistics", "fulfillment",
    "content writing", "editorial", "copywriting",
}

# ── Semantic Search Config ────────────────────────────────────────────────────
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 64

# ── JD Summary for Semantic Matching ─────────────────────────────────────────
JD_SUMMARY = """
Senior AI Engineer at a Series A AI-native talent intelligence platform.
Building ranking, retrieval, and matching systems for a recruiting platform.
Requires production experience with embeddings-based retrieval, vector databases,
hybrid search infrastructure, Python, and evaluation frameworks for ranking systems.
5-9 years experience, strong preference for product company backgrounds.
Must have shipped end-to-end ranking, search, or recommendation systems.
Experience with NLP, information retrieval, semantic search, LLM fine-tuning.
Located in India, preferably Pune or Noida. Hybrid work mode.
Looking for someone who combines deep ML technical depth with scrappy
product-engineering execution. Not a research role — must write production code.
"""

JD_KEYWORDS_FOR_EMBEDDING = """
AI engineer machine learning embeddings vector database retrieval ranking
NLP information retrieval semantic search sentence-transformers FAISS Pinecone
Weaviate Qdrant Milvus hybrid search BM25 Python PyTorch TensorFlow
deep learning neural networks transformer BERT GPT LLM fine-tuning LoRA
recommendation system search ranking NDCG MRR evaluation A/B testing
production deployment model serving MLOps data pipeline feature engineering
product company startup Series A recruiting HR-tech talent platform
"""
