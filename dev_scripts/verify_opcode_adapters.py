"""Bounded GPU acceptance of the real adapter Trainer/runtime, NOT a quality benchmark."""
from __future__ import annotations
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from pylingual.opcodes import OpcodeRegistry, OPCODE_FIELD, tokenizer_fingerprint
from pylingual.shared_models import load_version, load_family
from pylingual.models import VersionedTranslator
from pylingual.lora import adapter_key
from dev_scripts.assemble_shared_adapters import assemble
from dev_scripts.publish_shared_backbone import publish


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--task",choices=("statement","segmentation"),required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);root,out,task=args.root,args.output,args.task
    out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(root/"tokenizers"/task)
    registry=OpcodeRegistry.load(root/"tokenizers"/task/"opcode_registry.json",tokenizer)
    fixtures=[]
    with (root/"data-100k/accepted.jsonl").open() as stream:
        for line in stream:
            item=json.loads(line)
            if item["version"]!="3.14":continue
            with gzip.open(item["chunk"],"rt") as chunks:rows=json.load(chunks)[task]
            if rows:
                fixtures.append((item,rows))
            if len(fixtures)==4:break
    assert len(fixtures)==4
    for version in ("3.14","3.15"):
        train,valid=[],[]
        for index,(item,rows) in enumerate(fixtures):
            for row in rows[:8]:
                bytecode=row[1]
                if version=="3.15":
                    match=OPCODE_FIELD.match(bytecode.split(" <SEP> ")[0])
                    assert match is not None
                    begin,end=match.span(2)
                    bytecode=bytecode[:begin]+"SYNTHETIC_NEW_OP"+bytecode[end:]
                entry={"version":version,"file":item["source_sha512"],"source":row[0],"bytecode":bytecode}
                if task=="segmentation":entry["boundary"]=row[2]
                (valid if index==3 else train).append(entry)
        for label,rows in (("train",train),("valid",valid)):
            (out/f"{version}-{label}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    base=publish(root/"gpu-smoke-20261020-v3"/task/"checkpoint",root/"tokenizers"/task,task,out/"base",
                 training_versions=[f"3.{minor}" for minor in range(6,15)],smoke_only=True)
    command=[sys.executable,"-m","dev_scripts.train_shared_adapter","--family",str(base),"--steps","4","--batch-size","2","--accumulation","1","--rank","8","--checkpoint-steps","2"]
    for version in ("3.14","3.15"):
        subprocess.run(command+["--version",version,"--data",str(out/f"{version}-train.jsonl"),"--validation-data",str(out/f"{version}-valid.jsonl"),"--output",str(out/version)],check=True)
    checkpoint=out/"3.15/training/checkpoint-2"
    assert (checkpoint/"optimizer.pt").exists() and (checkpoint/"trainer_state.json").exists()
    subprocess.run(command+["--version","3.15","--data",str(out/"3.15-train.jsonl"),"--validation-data",str(out/"3.15-valid.jsonl"),"--output",str(out/"resumed"),"--resume",str(checkpoint)],check=True)
    assembled=assemble(base,[out/"3.14/adapters/3.14",out/"3.15/adapters/3.15"],out/"assembled")
    first=load_version(assembled,task,"3.14",torch.device("cuda:0"))
    second=load_version(assembled,task,"3.15",torch.device("cuda:0"))
    bank=second[0] if task=="statement" else second.bank
    assert bank is (first[0] if task=="statement" else first.bank)
    delta=bank.input_updates.rows[adapter_key("3.15")].delta
    assert delta.numel()>0 and bool(delta.detach().ne(0).any())
    original=json.loads((out/"3.15/adapters/3.15/adapter.json").read_text())
    resumed=json.loads((out/"resumed/adapters/3.15/adapter.json").read_text())
    a=torch.load(out/"3.15/adapters/3.15"/original["weights"],map_location="cpu",weights_only=True)
    b=torch.load(out/"resumed/adapters/3.15"/resumed["weights"],map_location="cpu",weights_only=True)
    errors={name:float((a[name]-b[name]).abs().max()) if a[name].numel() else 0 for name in a}
    assert max(errors.values())<1e-5,errors
    raw=json.loads((out/"3.15-train.jsonl").read_text().splitlines()[0])["bytecode"]
    if task=="segmentation":
        predictions=second([raw],batch_size=1)[0]
        assert len(predictions)==len(raw.split(" <SEP> "))
        runtime={"instruction_count":len(predictions)}
    else:
        translator=VersionedTranslator(second[0],second[1],torch.device("cuda:0"),"3.15","")
        translator.generation_kwargs["max_new_tokens"]=8
        translated=translator([raw])
        assert len(translated)==1 and "Decompiler error" not in translated[0],translated
        runtime={"translation_request_completed":True}
    report={"complete":True,"task":task,"smoke_only":True,"synthetic_future_opcode":True,
            "normal_trainer_fp16":True,"checkpoint_resume_max_abs_error":max(errors.values()),
            "new_token_delta_norm":float(delta.detach().norm()),"shared_backbone_reused":True,"runtime":runtime,
            "updates":{version:json.loads((out/version/"family.json").read_text())["adapter_updates"][version]["optimizer_updates"] for version in ("3.14","3.15")}}
    write(out/"result.json",report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()
