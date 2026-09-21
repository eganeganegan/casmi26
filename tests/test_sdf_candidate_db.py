import gzip

from rdkit import Chem

from casmi.retrieval import CandidateDatabase, build_sdf_candidate_database


def test_build_sdf_candidate_database_streams_gzip_and_deduplicates(tmp_path) -> None:
    plain = tmp_path / "source.sdf"
    writer = Chem.SDWriter(str(plain))
    for identifier, smiles in (("one", "CCO"), ("duplicate", "OCC"), ("two", "CCC")):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("chembl_id", identifier)
        writer.write(mol)
    writer.close()
    compressed = tmp_path / "source.sdf.gz"
    with plain.open("rb") as source, gzip.open(compressed, "wb") as destination:
        destination.write(source.read())
    output = tmp_path / "candidates.parquet"
    report = build_sdf_candidate_database(
        compressed,
        output,
        source="test",
        id_property="chembl_id",
        batch_size=1,
    )
    database = CandidateDatabase.from_parquet(output)
    assert report == {
        "input_records": 3,
        "output_rows": 2,
        "duplicates": 1,
        "invalid": 0,
    }
    assert len(database) == 2
    assert set(database.frame["candidate_id"].to_list()) == {"one", "two"}
