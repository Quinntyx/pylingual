"""Assemble independently trained version adapters into one immutable shared family."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from pylingual.shared_models import load_family


def assemble(family, adapters, output):
    family,output=Path(family),Path(output)
    if output.exists() or family.resolve() in output.resolve().parents:
        raise ValueError("Output must be a fresh directory outside the source family")
    bank,_,metadata,_=load_family(family,json.loads((family/"family.json").read_text())["task"],torch.device("cpu"))
    versions={}
    for directory in map(Path,adapters):
        adapter=json.loads((directory/"adapter.json").read_text());version=adapter["version"]
        if version in versions or version in metadata["adapters"]:
            raise ValueError("Duplicate adapter version; start from a backbone-only family")
        bank.load(version,directory)
        versions[version]=directory
    shutil.copytree(family,output,ignore=shutil.ignore_patterns("family.json"))
    updated=dict(metadata,complete=False)
    (output/"family.json").write_text(json.dumps(updated,indent=2)+"\n")
    updated["adapters"]=dict(metadata["adapters"])
    updated["opcode_bindings"]=dict(metadata.get("opcode_bindings",{}))
    for version,directory in versions.items():
        relative=f"adapters/{version}";shutil.copytree(directory,output/relative)
        updated["adapters"][version]=relative
        updated["opcode_bindings"][version]=bank.bindings[version]
    updated["versions"]=sorted(set(metadata["versions"])|set(versions),key=lambda x:tuple(map(int,x.split('.'))))
    updated["assembled_from"]={version:str(path.resolve()) for version,path in versions.items()}
    updated["complete"]=True
    temporary=output/"family.json.tmp";temporary.write_text(json.dumps(updated,indent=2)+"\n");temporary.replace(output/"family.json")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(assemble(args.family,args.adapter,args.output))
