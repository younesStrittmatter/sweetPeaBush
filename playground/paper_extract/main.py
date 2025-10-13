# main.py
from pathlib import Path
from dotenv import load_dotenv

from sweetExtract.project import Project
from sweetExtract.steps.find_data_links import FindDataLinks
from sweetExtract.steps.download_data import DownloadData
from sweetExtract.steps.extract_studies import ExtractStudies
from sweetExtract.steps.describe_experiments import DescribeExperiments
from sweetExtract.steps.unpack_data import BuildUnpackedData
from sweetExtract.steps.catalog import Catalog
from sweetExtract.steps.catalog_for_llm import CatalogForLLM
from sweetExtract.steps.llm_propose_trial_candidates import LLMProposeTrialCandidates
from sweetExtract.steps.expand_per_subject_candidates import ExpandPerSubjectCandidates
from sweetExtract.steps.probe_candidate_tables import ProbeCandidateTables
from sweetExtract.steps.llm_refine_trial_plan import LLMRefineTrialPlan
from sweetExtract.steps.execute_trial_split import ExecuteTrialSplit
from sweetExtract.steps.sb_schema_preview import SBTrialSchemaPreview
from sweetExtract.steps.sb_trial_schema_for_llm import SBTrialSchemaForLLM
from sweetExtract.steps.llm_map_trial_schema import LLMMapTrialSchema
from sweetExtract.steps.llm_build_sweetbean_plan import LLMBuildSweetBeanPlan




load_dotenv()

PROJECTS = [
    {"root": Path("./ZivonyEtEimer2021"), "pdf_path": Path("./papers/ZivonyEtEimer2021/paper.pdf")},
    {"root": Path("./WeindelEtAl2021"), "pdf_path": Path("./papers/WeindelEtAl2021/paper.pdf")},
    # {"root": Path("./JacobEtAl2021"), "pdf_path": Path("./papers/JacobEtAl2021/paper.pdf")},
    # {"root": Path("./BrehmEtMeyer2021"), "pdf_path": Path("./papers/BrehmEtMeyer2021/paper.pdf")},
    {"root": Path("./CraccoEtAl2022"), "pdf_path": Path("./papers/CraccoEtAl2022/paper.pdf")},
    # {"root": Path("./GhasemiEtAl2022"), "pdf_path": Path("./papers/GhasemiEtAl2022/paper.pdf")},
    # {"root": Path("./HughesEtAl2020"), "pdf_path": Path("./papers/HughesEtAl2020/paper.pdf")},
    # {"root": Path("./JEP_G/retrieve1"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve1.pdf")},
    # {"root": Path("./JEP_G/retrieve2"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve2.pdf")},
    # {"root": Path("./JEP_G/retrieve3"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve3.pdf")},
    # {"root": Path("./JEP_G/retrieve4"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve4.pdf")},
    # {"root": Path("./JEP_G/retrieve5"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve5.pdf")},
    {"root": Path("./JEP_G/retrieve6"), "pdf_path": Path("./papers/JEP_G/pdf/retrieve6.pdf")},
]

if __name__ == "__main__":
    for cfg in PROJECTS:
        p = Project(**cfg)

        p.add_steps([
            FindDataLinks(),  # runs only if data_raw is empty/missing (via should_run)
            DownloadData(),  # depends_on FindDataLinks; runs only if data_raw empty + sources present
            ExtractStudies(),  # you can add should_run to require data presence if you want
            DescribeExperiments(),  # depends_on ExtractStudies
            BuildUnpackedData(),
            Catalog(),
            CatalogForLLM(),
            LLMProposeTrialCandidates(),
            ExpandPerSubjectCandidates(),
            ProbeCandidateTables(),
            LLMRefineTrialPlan(),
            ExecuteTrialSplit(),
            SBTrialSchemaPreview(),
            SBTrialSchemaForLLM(),
            LLMMapTrialSchema(force=True),
            # LLMBuildSweetBeanPlan(force=True),
            # ValidateSweetBeanPlan(),
            # BuildTrialSchemaPreview(),  # scan a few CSVs per experiment
            # LLMSweetBeanSkeleton(),  # design-only timeline
            # LLMMapParametersToColumns(),  # bind parameters to columns/constants/mappings
            # ValidateSweetBeanMapping(),  # sanity-check bindings on real data
            # LLMResolveEncodings(),  # only meaningful if validation found gaps
            # AssembleSweetBeanPlan(),  # merged, code-ready JSON plan
            # IndexRawData(),  # should_run can require data_raw files
            # PlanParsers(),  # depends_on IndexRawData (+ whatever else you set)
            # InferExperimentLabels(),  # depends_on PlanParsers
            # RunParsers(),  # depends_on PlanParsers (+ maybe IndexRawData)
            # CleanupTrialTables(),  # depends_on RunParsers
        ])

        p.run_all()
