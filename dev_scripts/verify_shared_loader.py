"""Check public load_models dispatch using bounded, explicitly synthetic GPU fixtures."""
import argparse
import json
from pathlib import Path
from pylingual.models import load_models
from pylingual.editable_bytecode import PYCFile
from pylingual.decompiler import Decompiler
from pylingual.utils.version import PythonVersion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root
    paths={task:root/task/"assembled" for task in ("statement","segmentation")}
    # Runtime experiment fixture only: not personal or durable machine configuration.
    configuration={"SHARED_BACKBONES":{"STATEMENT_MODEL":str(paths["statement"]),"SEGMENTATION_MODEL":str(paths["segmentation"])}}
    config=root/"inference.json";config.write_text(json.dumps(configuration,indent=2)+"\n")
    versions={};native_stages={}
    for version in ("3.14","3.15"):
        segmenter,translator=load_models(config_file=config,version=PythonVersion(version))
        rows={task:json.loads((root/task/f"{version}-train.jsonl").read_text().splitlines()[0]) for task in paths}
        predictions=segmenter([rows["segmentation"]["bytecode"]],batch_size=1)[0]
        assert len(predictions)==len(rows["segmentation"]["bytecode"].split(" <SEP> "))
        translator.generation_kwargs["max_new_tokens"]=8
        translated=translator([rows["statement"]["bytecode"]])
        assert len(translated)==1 and "Decompiler error" not in translated[0]
        versions[version]=(segmenter,translator)
        pyc=PYCFile(root/f"native-{version}.pyc")
        assert str(pyc.version)==version
        decompiler=Decompiler(pyc,segmenter,translator,PythonVersion(version),top_k=1,timeout=60)
        decompiler.mask_bytecode()
        decompiler.run_segmentation()
        decompiler.run_translation()
        assert decompiler.translation_results
        native_stages[version]={"code_objects":len(decompiler.ordered_bytecodes),
                                "translated_statements":len(decompiler.source_lines)}
    assert versions["3.14"][0].bank is versions["3.15"][0].bank
    assert versions["3.14"][1].bank is versions["3.15"][1].bank
    # The no-adapter ablation must retain bytecode binding and atomic alignment.
    configuration["SHARED_BACKBONES"]["USE_ADAPTERS"]=False
    ablation=root/"backbone-only-inference.json";ablation.write_text(json.dumps(configuration)+"\n")
    segmenter,translator=load_models(config_file=ablation,version=PythonVersion("3.15"))
    assert segmenter.version is None and translator.version is None
    assert "<future_opcode_00>" in translator._prepare_bytecode(rows["statement"]["bytecode"])
    assert len(segmenter([rows["segmentation"]["bytecode"]],batch_size=1)[0])==len(rows["segmentation"]["bytecode"].split(" <SEP> "))
    result={"complete":True,"smoke_only":True,"synthetic_future_opcode":True,
            "public_load_models":True,"versions":["3.14","3.15"],
            "one_backbone_per_task":True,"backbone_only_ablation":True,"native_decompiler_stages":native_stages}
    (root/"public-loader-result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
