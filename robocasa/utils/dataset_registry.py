from collections import OrderedDict
from copy import deepcopy
import os
from pathlib import Path

import robocasa


SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    PnPCounterToCab=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/c0a0vdyqeqh6o9z4c57sk61aypladizj.hdf5",
            human_im="https://utexas.box.com/shared/static/gznii250ip99731ii2r0ml6be7gzt2cp.hdf5",
        ),
    ),
    PnPCabToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/fgd165p91ts9qurfb496ubi1xdqgrz97.hdf5",
            human_im="https://utexas.box.com/shared/static/pdfh88slq4bazfglixaw80whvbtlbw6r.hdf5",
        ),
    ),
    PnPCounterToSink=dict(
        horizon=700,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/nc0x55xwlgs4acj97ngdj1q6pv450ooo.hdf5",
            human_im="https://utexas.box.com/shared/static/0z1wr8iwfusfqqpp56j3vfxvvse1x0vc.hdf5",
        ),
    ),
    PnPSinkToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/uyhepwjyr0880a5q03fc61qmhbrhwy7k.hdf5",
            human_im="https://utexas.box.com/shared/static/pe03k5qvcuq1nzbpfag5pfo2t8fzvl9i.hdf5",
        ),
    ),
    PnPCounterToMicrowave=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/z1qwxsuczjmv68p7267nylbv9ebc6g4p.hdf5",
            human_im="https://utexas.box.com/shared/static/2kun70ehqzanl5h0hbgpdezrfrvuimq8.hdf5",
        ),
    ),
    PnPMicrowaveToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/emnl5i3s621sf5smgek5lu4twbc326jt.hdf5",
            human_im="https://utexas.box.com/shared/static/z2t1kyto32thprw7xvq3r4tjvkdz0dl2.hdf5",
        ),
    ),
    PnPCounterToStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/emnl5i3s621sf5smgek5lu4twbc326jt.hdf5",
            human_im="https://utexas.box.com/shared/static/z2t1kyto32thprw7xvq3r4tjvkdz0dl2.hdf5",
        ),
    ),
    PnPStoveToCounter=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/1m3v1rkh7gna5ujt0vcwzej7stv5rdio.hdf5",
            human_im="https://utexas.box.com/shared/static/rht9bfh38gaue5ig17ubpsibb4jrgtyv.hdf5",
        ),
    ),
    OpenSingleDoor=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/hjukxdpqvnrghtbjbhin28idzkysq7oa.hdf5",
            human_im="https://utexas.box.com/shared/static/vhvlbuza3g1lpqa9m6zvmbzh8xquvqy8.hdf5",
        ),
    ),
    CloseSingleDoor=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/gea28p5mrvj9lnw1s6c8fqvzxk5f779i.hdf5",
            human_im="https://utexas.box.com/shared/static/2wnm0u1x9fp9ni02pmzqzhpjsb1kgfrr.hdf5",
        ),
    ),
    OpenDoubleDoor=dict(
        horizon=1000,
        human_path="v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/xl23utiszzdbjdqlveq1mxxca9m3kbvw.hdf5",
            human_im="https://utexas.box.com/shared/static/8swihowjd5fdk1vpf0k94gl72f8nbjeb.hdf5",
        ),
    ),
    CloseDoubleDoor=dict(
        horizon=700,
        human_path="v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zozyctzejc2nrlpjwqjq4w761830kq45.hdf5",
            human_im="https://utexas.box.com/shared/static/14f2ssfhwfhyo9cvj0s3kq303bv38g6t.hdf5",
        ),
    ),
    OpenDrawer=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/990d2f339zvalvlw50s4s1l6sfvt018m.hdf5",
            human_im="https://utexas.box.com/shared/static/d8a0g5827kbm4ufk3p1tbuz3idstgum8.hdf5",
        ),
    ),
    CloseDrawer=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ooekd1zdy02hfu234xm5h63f7mzbez4b.hdf5",
            human_im="https://utexas.box.com/shared/static/4r5w0a6i4jtgv5qmqx09fnqh5d7c45oi.hdf5",
        ),
    ),
    TurnOnSinkFaucet=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/7vogk85ed3sm1o1vo8fzast9rs8zshgk.hdf5",
            human_im="https://utexas.box.com/shared/static/f0brygtzwgmmwccg58b6m82yapa4jnji.hdf5",
        ),
    ),
    TurnOffSinkFaucet=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ukkycoa2c6k5d6bpt1cdplzk2nrg879t.hdf5",
            human_im="https://utexas.box.com/shared/static/ceewfn4ydhprupdcdppfe8wu4x61oxdg.hdf5",
        ),
    ),
    TurnSinkSpout=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_sink/TurnSinkSpout/2024-04-29",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zxop70u9s6rl4udz7fhp4besrvu508pj.hdf5",
            human_im="https://utexas.box.com/shared/static/nnojt3760k143fxwhf2u0kzh1i56n7x1.hdf5",
        ),
    ),
    TurnOnStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/ow4npc933yty4blxg8rc2pgjp6f1yc36.hdf5",
            human_im="https://utexas.box.com/shared/static/dewbngq5wk6dipb29x6984x6dygu2gck.hdf5",
        ),
    ),
    TurnOffStove=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/vp6u3huv9c2vg5r09105gkbmwguycnn0.hdf5",
            human_im="https://utexas.box.com/shared/static/8tukea09szcjb2ncbe43zt65n7dzg4eh.hdf5",
        ),
    ),
    CoffeeSetupMug=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/zxytzos86xes7jo8z0gp7lklawbiyo8g.hdf5",
            human_im="https://utexas.box.com/shared/static/pv2i49t4p7238gp34txhm7jcl7domw58.hdf5",
        ),
    ),
    CoffeeServeMug=dict(
        horizon=600,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/tse3mkxx913pf4d4ij7fplchtv1rchkq.hdf5",
            human_im="https://utexas.box.com/shared/static/ts3537f93dzjpkux19syy0pndw5re231.hdf5",
        ),
    ),
    CoffeePressButton=dict(
        horizon=300,
        human_path="v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/t6oblmeilexe9ndca4ccaxdclw1tzps5.hdf5",
            human_im="https://utexas.box.com/shared/static/l5dnmcfd0r36vhdqgjchxo20vajt7ohl.hdf5",
        ),
    ),
    TurnOnMicrowave=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/o1sp7qrcd97b6jo68olx58l35wffpbna.hdf5",
            human_im="https://utexas.box.com/shared/static/t6eromogy5is2no3s1e5bk1nb2hgab2k.hdf5",
        ),
    ),
    TurnOffMicrowave=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/m2exfytqjd0496y70b3ou2i8993ryb50.hdf5",
            human_im="https://utexas.box.com/shared/static/0drm2h7fgd5857x8xgcj1lph23srpbj1.hdf5",
        ),
    ),
    NavigateKitchen=dict(
        horizon=500,
        human_path="v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/pzvzuvmkla74kro84yw13uwjr2g89z8m.hdf5",
            human_im="https://utexas.box.com/shared/static/mbi6011svy2zb436wsf2hhc337aduwmw.hdf5",
        ),
    ),
)


MULTI_STAGE_TASK_DATASETS = OrderedDict(
    ArrangeVegetables=dict(
        human_path="v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11",
        horizon=1200,
        activity="chopping_food",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/hiwbv68h3vty9ksqsdhcw2xhuvkwcf3o.hdf5",
            human_im="https://utexas.box.com/shared/static/ovzt0bod4nrrack6dak12jn1fx2cnwbw.hdf5",
        ),
    ),
    MicrowaveThawing=dict(
        human_path="v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11",
        horizon=1000,
        activity="defrosting_food",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/wreambog4jak8og9nbhm0nr2t6w5ygdt.hdf5",
            human_im="https://utexas.box.com/shared/static/ehwo6728r3qyi8g0h70zhawijojqotnm.hdf5",
        ),
    ),
    RestockPantry=dict(
        human_path="v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10",
        horizon=1000,
        activity="restocking_supplies",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/t2g6rdyh006bw09cxxxamd7thton5grl.hdf5",
            human_im="https://utexas.box.com/shared/static/35yr2xmsmwq0xkknvlmlivh9ifste71q.hdf5",
        ),
    ),
    PreSoakPan=dict(
        human_path="v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10",
        horizon=1500,
        activity="washing_dishes",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/d4vg2pag7465xp99vej1skk3frmohzk1.hdf5",
            human_im="https://utexas.box.com/shared/static/etotr8tn0z57ttb311bibh0mdmc65j96.hdf5",
        ),
    ),
    PrepareCoffee=dict(
        human_path="v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07",
        horizon=1000,
        activity="brewing",
        download_links=dict(
            human_raw="https://utexas.box.com/shared/static/qhtle2ccvm17g65ecg2vrudpkl9v5wkk.hdf5",
            human_im="https://utexas.box.com/shared/static/qq3p4ctlmiv751jx5av92aaz9dxylw76.hdf5",
        ),
    ),
)

def get_ds_path(task, ds_type, return_info=False):
    if task in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[task]
    elif task in MULTI_STAGE_TASK_DATASETS:
        ds_config = MULTI_STAGE_TASK_DATASETS[task]
    else:
        raise ValueError
    
    if ds_type == "human_raw":
        folder = ds_config["human_path"]
        fname = "demo.hdf5"
    elif ds_type == "human_im":
        folder = ds_config["human_path"]
        if task in SINGLE_STAGE_TASK_DATASETS:
            fname = "demo_gentex_im128_randcams.hdf5"
        elif task in MULTI_STAGE_TASK_DATASETS:
            fname = "demo_gentex_im128.hdf5"
    else:
        raise ValueError

    base_dir = Path(robocasa.__path__[0]).parent.absolute()
    ds_path = os.path.join(os.path.join(base_dir, "datasets"), folder, fname)

    if return_info is False:
        return ds_path

    ds_info = {}
    ds_info["url"] = ds_config["download_links"][ds_type]
    return ds_path, ds_info
