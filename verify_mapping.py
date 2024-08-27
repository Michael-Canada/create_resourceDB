import os
import io
from typing import Dict, NamedTuple
import requests

import pandas as pd

_MISO_MANUAL_MAPPING = {
    # plant_id,unit_id,utility_id
    "DEUELHAR DEUEL_GRE_H2": [(62943, "GEN1", 17650)],
    "GRJ_AMRN GRJ_AND_OGDEN": [(1143, "2", 17650)],
    "MINDNWF  BTURTLE2": [(65143, "BTWF2", 57480)],
    "NOPS     G1": [(60928, "RICE1", 13478)],
    "NOPS     G2": [(60928, "RICE2", 13478)],
    "NOPS     G3": [(60928, "RICE3", 13478)],
    "NOPS     G4": [(60928, "RICE4", 13478)],
    "NOPS     G5": [(60928, "RICE5", 13478)],
    "NOPS     G6": [(60928, "RICE6", 13478)],
    "NOPS     G7": [(60928, "RICE7", 13478)],
    "PRAIR_ST UN2_MEUC": [(55856, "PC2", 15330)],
    "TOWNELIN TOWNE_ST1_2": [(55641, "STG1", 20856)],
    "WILLCO   LOCKPHYD": [(10903, "1GEN", 3473), (10903, "2GEN", 3473)],
    "HARDIN   HRDN23SP": [(63029, "GEN1", 49893)],
    "HARDIN   HRDN34SP": [(63029, "GEN1", 49893)],
    "HARDIN   HRDN12SP": [(63828, "GEN1", 49893)],
}

_SPP_MANUAL_MAPPING = {
    "HITCH 345 197 GPWC UN": [(64665, "GR01", 64260)],
    "HITCH 345 198 GPWB UN": [(64665, "GR01", 64260)],
    "HITCH 345 199 GPVW UN": [(64665, "GR01", 64260)],
    "RIV453 16 R12 12 UN": [(1239, "12-2", 5860)],
    "WESTPRK 69 GT1 HURON_GT1 UN": [(3344, "2A", 13809)],
    "WESTPRK 69 GT2 HURON_GT2 UN": [(3344, "2A", 13809)],
    "WESTPRK 69 GT3 HURON_GT3 UN": [(3344, "2A", 13809)],
    "WESTPRK 69 GT4 HURON_GT4 UN": [(3344, "2A", 13809)],
    "WESTPRK 69 GT5 HURON_GT5 UN": [(3344, "2A", 13809)],
    "WESTPRK 69 GT6 HURON_GT6 UN": [(3344, "2A", 13809)],
    "BLK_DOG 18 9002 BLK_DO_6_UNIT UN": [(1904, "6-1", 13781)],
}

_ISO_TO_MANUAL_MAPPING = {"MISO": _MISO_MANUAL_MAPPING, "SPP": _SPP_MANUAL_MAPPING}


def _get_manual_mapping(iso: str) -> pd.DataFrame:
    manual_mapping = _ISO_TO_MANUAL_MAPPING[iso]
    result = []

    for uid, gens in manual_mapping.items():
        result += [
            {
                "generator_name": uid,
                "eia_utility_id": gen[2],
                "eia_plant_code": gen[0],
                "eia_gen_id": gen[1],
            }
            for gen in gens
        ]
    return pd.DataFrame(result)


def _override_with_manual_mapping(mapping: pd.DataFrame, iso: str) -> pd.DataFrame:
    manual_mapping = _get_manual_mapping(iso)
    result = mapping[~mapping.generator_name.isin(manual_mapping.generator_name)].copy(
        deep=True
    )
    result = pd.concat([result, manual_mapping], ignore_index=True)
    return result


_MISO_STUDY_ZONES = {
    "AEWC",
    "ALT ALTE",
    "ALT ALTW",
    "AMI ATXI",
    "AMI CILC",
    "AMI CIPS",
    "AMI IP",
    "AMI SOYL",
    "AMM UE",
    "BRE BREC",
    "CIN CIN",
    "CLAY",
    "CLE CLEC",
    "CON CONS",
    "CWL CWLD",
    "CWL CWLP",
    "DEAU",
    "DEC DECO",
    "DEVI",
    "DPC DPC",
    "EAI APL",
    "EES ELA",
    "EES ENOI",
    "EES ETI",
    "EMB EMI",
    "FREE",
    "GLH",
    "GRE GRE",
    "HE  HE",
    "HLLD",
    "HMP HMPL",
    "IPL IPL",
    "KMJX",
    "LAF LAFA",
    "LAG BEC",
    "LAG CLA",
    "LAG CON",
    "LAG JDC",
    "LAG LAGN",
    "LAG NEL",
    "LAG PCP",
    "LAG SLC",
    "LAG SLM",
    "LAG WST",
    "LEP LEPA",
    "MDU MDU",
    "MEC MEC",
    "MGE MGE",
    "MIU MIUP",
    "MP  MP",
    "MPU",
    "MPW MPW",
    "NEOG",
    "NIP NIPS",
    "NSP NSP",
    "OTP OTP",
    "SIG SIGE",
    "SIP SIPC",
    "SME BBA",
    "SME SME",
    "SMP SMP",
    "UPP UPPC",
    "WEC WEC",
    "WPS WPS",
    # EXTERNAL ZONES
    "WAU WAUE",
    "KCP KCPL",
    "CSW SWEP",
    "SPA SPA",
    "AEP AEP",
    "MPS MPS",
    "OPP OPPD",
    "FE  FE",
    "CE  CE",
    "AEC AECI",
}

_SPP_STUDY_ZONES = {
    "CSW CSWS",
    "EDE EDE",
    "GRD GRDA",
    "IND INDN",
    "KAC KACY",
    "KCP KCPL",
    "LES LES",
    "MPS MPS",
    "NPP NPPD",
    "OKG OKGE",
    "OPP OPPD",
    "SEC SECI",
    "SPR SPRM",
    "SPS SPS",
    "WAU WAUE",
    "WFE WFEC",
    "WR  WR",
    "SPA SPA",
    # External Zones
    "AEC AECI",
    "AEC GRDX",
    "MDU MDU",
    "NSP NSP",
}

_ISO_TO_STUDY_ZONES = {"MISO": _MISO_STUDY_ZONES, "SPP": _SPP_STUDY_ZONES}


def _remove_space(s: str) -> str:
    return s.replace(" ", "")


def _map_to_case_generator_name(s: str, case_gen_names_map: Dict[str, str]) -> str:
    if _remove_space(s) not in case_gen_names_map:
        print(f'Generator name "{s}" not found in case generators')
        return s
    if case_gen_names_map[_remove_space(s)] != s:
        print(
            f'Generator name "{s}" mapped to "{case_gen_names_map[_remove_space(s)]}"'
        )
    return case_gen_names_map[_remove_space(s)]


class MappingData(NamedTuple):
    generator_name: str
    eia_utility_id: int
    eia_plant_code: int
    eia_gen_id: str


def test_clean_generator_mapping() -> None:
    mapping_data = pd.DataFrame(
        {
            "generator_name": ["Gen 1", "Gen 2", "Gen 3", "Gen 4"],
            "eia_utility_id": [1, 2, 3, 4],
            "eia_plant_code": [1, 2, 3, 4],
            "eia_gen_id": ["1", "02", "3 ", " 4 "],
        }
    )
    case_gens = pd.DataFrame(
        {
            "uid": ["Gen 1", "Gen  2", "Gen   3", "Gen4"],
            "latest_zone": ["zone1", "zone2", "zone3", "zone4"],
        }
    )
    eia_data = pd.DataFrame(
        {"plant_id": [1, 2, 3, 4], "generator_id": ["1", "2", "03", "0004"]}
    )

    new_mapping = clean_generator_mapping(mapping_data, case_gens, eia_data)

    assert new_mapping.generator_name.to_list() == [
        "Gen 1",
        "Gen  2",
        "Gen   3",
        "Gen4",
    ]
    assert new_mapping.eia_gen_id.to_list() == ["1", "2", "03", "0004"]


def clean_generator_mapping(
    mapping_data: pd.DataFrame,
    case_gens: pd.DataFrame,
    eia_data: pd.DataFrame,
    iso: str,
) -> pd.DataFrame:
    mapping_gens = [MappingData(**dict(elem)) for _, elem in mapping_data.iterrows()]
    eia_gens = list(zip(eia_data.plant_id, eia_data.generator_id))
    case_gen_names_map = {_remove_space(elem): elem for elem in case_gens.uid}
    gen_name_to_zone = dict(zip(case_gens.uid, case_gens.latest_zone))
    eia_gens_set = set(eia_gens)

    result = [elem for elem in mapping_gens]

    for i, elem in enumerate(mapping_gens):
        gen_name = _map_to_case_generator_name(elem.generator_name, case_gen_names_map)
        result[i] = result[i]._replace(generator_name=gen_name)
        unit_id = elem.eia_gen_id

        if (elem.eia_plant_code, unit_id) not in eia_gens_set:
            # Check if unit id has an extra leading zero
            if (elem.eia_plant_code, unit_id.strip().lstrip("0")) in eia_gens_set:
                new_unit_id = unit_id.strip().lstrip("0")
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id has two extra leading zero
            elif (elem.eia_plant_code, unit_id.strip().lstrip("00")) in eia_gens_set:
                new_unit_id = unit_id.strip().lstrip("00")
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id has three extra leading zero
            elif (elem.eia_plant_code, unit_id.strip().lstrip("000")) in eia_gens_set:
                new_unit_id = unit_id.strip().lstrip("000")
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id is missing a leading zero
            elif (elem.eia_plant_code, "0" + unit_id.strip()) in eia_gens_set:
                new_unit_id = "0" + unit_id.strip()
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id is missing two leading zero
            elif (elem.eia_plant_code, "00" + unit_id.strip()) in eia_gens_set:
                new_unit_id = "00" + unit_id.strip()
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id is missing three leading zero
            elif (elem.eia_plant_code, "000" + unit_id.strip()) in eia_gens_set:
                new_unit_id = "000" + unit_id.strip()
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            # Check if unit id is leading or trailing space
            elif (elem.eia_plant_code, unit_id.strip()) in eia_gens_set:
                new_unit_id = unit_id.strip()
                result[i] = result[i]._replace(eia_gen_id=new_unit_id)
                print(
                    f'Unit_id "{elem.eia_gen_id}" for generator "{elem.generator_name}" mapped to "{new_unit_id}"'
                )

            else:
                # if gen_name_to_zone[elem.generator_name] in _ISO_TO_STUDY_ZONES[iso]:
                print(
                    f'Error: Zone "{gen_name_to_zone[elem.generator_name]}" Generator "{gen_name}"'
                    + f' mapping data: (plant_id "{elem.eia_plant_code}", unit_id "{elem.eia_gen_id}"'
                    + f") not found in eia generators"
                )

    return pd.DataFrame(
        {
            "generator_name": [elem.generator_name for elem in result],
            "eia_utility_id": [elem.eia_utility_id for elem in result],
            "eia_plant_code": [elem.eia_plant_code for elem in result],
            "eia_gen_id": [elem.eia_gen_id for elem in result],
        }
    )


def _get_collection(iso: str) -> str:
    if iso == "MISO":
        COLLECTION = "miso-se"
    elif iso == "SPP":
        COLLECTION = "spp-se"
    else:
        raise ValueError("Invalid ISO")
    return COLLECTION


def _get_additional_mapping(iso: str) -> pd.DataFrame:
    if iso == "MISO":
        additional_mapping = pd.read_csv("./miso 1.csv")
    elif iso == "SPP":
        additional_mapping = pd.read_csv("./spp 1.csv")
    else:
        raise ValueError("Invalid ISO")

    additional_mapping = additional_mapping.rename(
        columns={
            "entity_id": "eia_utility_id",
            "plant_id": "eia_plant_code",
            "generator_id": "eia_gen_id",
        }
    )
    return additional_mapping


def get_new_mapping(
    iso: str, use_additional_mapping: bool = False, use_mannual_override: bool = True
) -> pd.DataFrame:
    def _get_auth(env_var: str = "SELF"):
        return tuple(os.environ[env_var].split(":"))

    AUTH = _get_auth()

    def _get_dfm(url, auth=AUTH):
        resp = requests.get(url, auth=auth)

        if resp.status_code != 200:
            print(resp.text)
            resp.raise_for_status()

        dfm = pd.read_csv(io.StringIO(resp.text))

        return dfm

    iso = iso.upper()
    COLLECTION = _get_collection(iso)

    df_eia = _get_dfm(
        "https://api1.marginalunit.com/misc-data/eia/generators/monthly?columns=plant_id,plant_name,plant_state,generator_id,energy_source_code,prime_mover_code,operating_month,operating_year,latitude,longitude,retirement_month,retirement_year,planned_operation_month,planned_operation_year,net_summer_capacity_mw"
    )
    df_eia.generator_id = df_eia.generator_id.astype(str)

    df_case_gen = _get_dfm(
        f"https://api1.marginalunit.com/reflow/{COLLECTION}/generators"
    )

    df_mapping = _get_dfm(
        f"https://api1.marginalunit.com/rms/eia/generators/{COLLECTION}/generator-mappings"
    )

    if use_additional_mapping:
        additional_mapping = _get_additional_mapping(iso)
        df_mapping = df_mapping[
            ~df_mapping.generator_name.isin(additional_mapping.generator_name)
        ].copy(deep=True)
        df_mapping = pd.concat([df_mapping, additional_mapping], ignore_index=True)

    df_mapping.eia_gen_id = df_mapping.eia_gen_id.astype(str)

    new_mapping = clean_generator_mapping(df_mapping, df_case_gen, df_eia, iso)
    # new_mapping.to_csv("~/downloads/" + iso.lower() + "_mapping_existing.csv")
    if use_mannual_override:
        new_mapping = _override_with_manual_mapping(new_mapping, iso)
        # new_mapping.to_csv("~/downloads/" + iso.lower() + "_add_manual_mapping.csv")
    return new_mapping


def main():
    get_new_mapping(iso="SPP", use_additional_mapping=True, use_mannual_override=True)


if __name__ == "__main__":
    main()
