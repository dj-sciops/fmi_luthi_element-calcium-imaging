import importlib
import inspect
import shutil
import pathlib
import pickle
from collections.abc import Callable
from datetime import datetime, timezone
import re
import os

import datajoint as dj
import numpy as np
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory
from element_interface.utils import memoized_result

from . import scan

logger = dj.logger

schema = dj.schema()

imaging = None


def activate(
    schema_name,
    *,
    imaging_module,
    create_schema=True,
    create_tables=True,
):
    """
    activate(schema_name, *, imaging_module, create_schema=True, create_tables=True)
        :param schema_name: schema name on the database server to activate the `field_processing` schema
        :param imaging_module: the activated imaging element for which this `processing` schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    global imaging
    imaging = imaging_module
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=imaging.__dict__,
    )
    imaging.Processing.key_source -= PreProcessing.key_source.proj()


# ---------------- Multi-field Processing (per-field basis) ----------------


@schema
class PreProcessing(dj.Computed):
    definition = """
    -> imaging.ProcessingTask
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration    
    """

    class Field(dj.Part):
        definition = """
        -> master
        -> scan.ScanInfo.Field
        ---
        params: longblob  # parameter set for this run
        processing_output_dir: varchar(1000)  #  Output directory of the processed scan relative to root data directory
        """

    class File(dj.Part):
        definition = """
        -> master.Field
        file_name: varchar(255)
        ---
        file: filepath@imaging-processed
        """

    @property
    def key_source(self):
        """
        Find ProcessingTask entries with method = "suite2p" and roi > 0 or method = "caimain" and depths > 1
        """
        ks = (
            imaging.ProcessingTask
            * scan.ScanInfo.proj("nrois", "nfields")
            * imaging.ProcessingParamSet.proj("processing_method")
            & "task_mode = 'trigger'"
        ) & "nfields >= 1"
        ks &= "(processing_method = 'suite2p' AND nrois > 0) OR (processing_method = 'caiman' AND nrois = 0)"
        return ks - imaging.Processing.proj()

    def make(self, key):
        execution_time = datetime.now(timezone.utc)
        processed_root_data_dir = scan.get_processed_root_data_dir()

        output_dir = (imaging.ProcessingTask & key).fetch1("processing_output_dir")

        if not output_dir:
            output_dir = imaging.ProcessingTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update processing_output_dir
            imaging.ProcessingTask.update1(
                {**key, "processing_output_dir": output_dir.as_posix()}
            )

        try:
            output_dir = find_full_path(processed_root_data_dir, output_dir)
        except FileNotFoundError:
            output_dir = processed_root_data_dir / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        method, params = (
            imaging.ProcessingTask * imaging.ProcessingParamSet & key
        ).fetch1("processing_method", "params")
        acq_software = (scan.Scan & key).fetch1("acq_software")

        field_ind = (scan.ScanInfo.Field & key).fetch("field_idx")
        sampling_rate, ndepths, nchannels, nfields, nrois = (
            scan.ScanInfo & key
        ).fetch1("fps", "ndepths", "nchannels", "nfields", "nrois")

        if method == "caiman" and acq_software == "PrairieView":
            from element_interface.prairie_view_loader import (
                PrairieViewMeta,
            )

            image_file = (scan.ScanInfo.ScanFile & key).fetch("file_path", limit=1)[0]
            pv_dir = find_full_path(scan.get_imaging_root_data_dir(), image_file).parent
            PVmeta = PrairieViewMeta(pv_dir)

            channel = (
                params.get("channel_to_process", 0)
                if PVmeta.meta["num_channels"] > 1
                else PVmeta.meta["channels"][0]
            )
            prepared_input_dir = output_dir.parent / "prepared_input"
            prepared_input_dir.mkdir(exist_ok=True)

            @memoized_result(
                uniqueness_dict=params,
                output_directory=prepared_input_dir,
            )
            def _run_write_bigtiff():
                _field_processing_tasks = []
                for field_idx, plane_idx in zip(
                    field_ind, PVmeta.meta["plane_indices"]
                ):
                    pln_output_dir = output_dir / f"pln{plane_idx}_chn{channel}"
                    pln_output_dir.mkdir(parents=True, exist_ok=True)

                    image_files = PVmeta.write_single_bigtiff(
                        plane_idx=plane_idx,
                        channel=channel,
                        output_dir=prepared_input_dir,
                        caiman_compatible=True,
                        overwrite=True,
                        gb_per_file=4,
                    )

                    _field_processing_tasks.append(
                        {
                            **key,
                            "field_idx": field_idx,
                            "params": {
                                **params,
                                "extra_dj_params": {
                                    "channel": channel,
                                    "plane_idx": plane_idx,
                                    "image_files": [
                                        f.relative_to(
                                            processed_root_data_dir
                                        ).as_posix()
                                        for f in image_files
                                    ],
                                },
                            },
                            "processing_output_dir": pln_output_dir.relative_to(
                                processed_root_data_dir
                            ).as_posix(),
                        }
                    )
                return _field_processing_tasks

            field_processing_tasks = _run_write_bigtiff()
        else:
            raise NotImplementedError(
                f"Field processing for {acq_software} scans with {method} is not yet supported in this table."
            )

        exec_dur = (datetime.now(timezone.utc) - execution_time).total_seconds() / 3600
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": exec_dur,
            }
        )
        self.Field.insert(field_processing_tasks)
        # Insert result files
        file_entries = []
        for field in field_processing_tasks:
            for f in field["params"]["extra_dj_params"]["image_files"]:
                f = processed_root_data_dir / f
                file_entries.append(
                    {
                        **key,
                        "field_idx": field["field_idx"],
                        "file_name": f.relative_to(prepared_input_dir).as_posix(),
                        "file": f,
                    }
                )
        # logger.info("Inserting %d file entries...", len(file_entries))
        # self.File.insert(file_entries)


@schema
class FieldMotionCorrection(dj.Computed):
    definition = """
    -> PreProcessing.Field
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    mc_params: longblob  # parameter set for this run
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@imaging-processed
        """

    def make(self, key):
        execution_time = datetime.now(timezone.utc)
        processed_root_data_dir = scan.get_processed_root_data_dir()

        output_dir, params = (PreProcessing.Field & key).fetch1(
            "processing_output_dir", "params"
        )
        extra_params = params.pop("extra_dj_params", {})
        output_dir = find_full_path(processed_root_data_dir, output_dir)

        acq_software = (scan.Scan & key).fetch1("acq_software")
        method = (imaging.ProcessingParamSet * imaging.ProcessingTask & key).fetch1(
            "processing_method"
        )
        sampling_rate = (scan.ScanInfo & key).fetch1("fps")

        if acq_software == "PrairieView" and method == "caiman":
            import multiprocessing
            import tifffile
            import caiman as cm
            from caiman.motion_correction import MotionCorrect
            from caiman.source_extraction.cnmf.cnmf import CNMF
            from caiman.source_extraction.cnmf.params import CNMFParams

            file_paths = [
                find_full_path(processed_root_data_dir, f)
                for f in extra_params["image_files"]
            ]

            images = []
            for f in file_paths:
                with tifffile.TiffFile(f) as tffl:
                    # downsample by 1000x in time
                    images.append(
                        tffl.asarray(key=np.arange(0, len(tffl.pages), 1000)).transpose(
                            1, 2, 0
                        )
                    )

            rho = cm.summary_images.local_correlations(np.dstack(images))
            half_median_correlation = np.nanmedian(rho) / 2

            logger.info("Min correlation set to: %f", half_median_correlation)
            params["min_corr"] = half_median_correlation

            # run caiman motion correction
            params["is3D"] = False
            params["fnames"] = [f.as_posix() for f in file_paths]
            params["fr"] = sampling_rate

            if "indices" in params:
                mc_indices = params.pop(
                    "indices"
                )  # Indices that restrict FOV for motion correction.
                params["motion"] = {
                    **params.get("motion", {}),
                    "indices": (slice(*mc_indices[0]), slice(*mc_indices[1])),
                }
            else:
                mc_indices = None

            output_dir = output_dir / "motion_correction"
            output_dir.mkdir(parents=True, exist_ok=True)

            @memoized_result(
                uniqueness_dict=params,
                output_directory=output_dir,
            )
            def _run_motion_correction():
                caiman_temp = os.environ.get("CAIMAN_TEMP")
                os.environ["CAIMAN_TEMP"] = str(output_dir)

                # use 60% of available cores (this step should be low memory)
                n_processes = np.floor(multiprocessing.cpu_count() * 0.6)
                n_processes = int(os.getenv("CAIMAN_MC_N_PROCESSES", n_processes))
                _, dview, n_processes = cm.cluster.setup_cluster(
                    backend="multiprocessing",
                    n_processes=n_processes,
                    maxtasksperchild=1,
                )
                try:
                    opts = CNMFParams(params_dict=params)
                    cnm = CNMF(n_processes, params=opts, dview=dview)
                    fnames = cnm.params.get("data", "fnames")
                    mc = MotionCorrect(fnames, dview=cnm.dview, **cnm.params.motion)
                    mc_base_attrs = list(mc.__dict__)

                    logger.info("Starting motion correction (CaImAn)...")
                    # only save F-order mmap if no cropping
                    mc.motion_correct(save_movie=mc_indices is None)
                    mc_results = {
                        k: v for k, v in mc.__dict__.items() if k not in mc_base_attrs
                    }

                    if cnm.params.get("motion", "pw_rigid"):
                        mc_results["b0"] = np.ceil(
                            np.maximum(
                                np.max(np.abs(mc.x_shifts_els)),
                                np.max(np.abs(mc.y_shifts_els)),
                            )
                        ).astype(int)
                        if cnm.params.get("motion", "is3D"):
                            cnm.estimates.shifts = [
                                mc.x_shifts_els,
                                mc.y_shifts_els,
                                mc.z_shifts_els,
                            ]
                        else:
                            cnm.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els]
                    else:
                        mc_results["b0"] = np.ceil(
                            np.max(np.abs(mc.shifts_rig))
                        ).astype(int)
                        cnm.estimates.shifts = mc.shifts_rig
                except Exception as e:
                    dview.terminate()
                    raise e
                else:
                    cm.stop_server(dview=dview)
                    logger.info("Motion correction (CaImAn) complete. Saving results.")
                    base_name = pathlib.Path(fnames[0]).stem
                    cnmf_mc_output_file = output_dir / (base_name + "_cnm_mc.hdf5")
                    cnm.save(cnmf_mc_output_file.as_posix())
                    if caiman_temp is not None:
                        os.environ["CAIMAN_TEMP"] = caiman_temp
                    else:
                        del os.environ["CAIMAN_TEMP"]
                extra_dj_params = {
                    "cnmf_mc_output_file": cnmf_mc_output_file.relative_to(
                        processed_root_data_dir
                    ).as_posix(),
                    "mc_results": mc_results,
                }
                return extra_dj_params

            extra_dj_params = _run_motion_correction()
            params["fnames"] = [
                f.relative_to(processed_root_data_dir).as_posix() for f in file_paths
            ]
            motion_correction_results = extra_dj_params.pop("mc_results")
            params["extra_dj_params"] = extra_dj_params
            pickle.dump(
                motion_correction_results,
                output_dir
                / f"{key['subject']}_session{key['session']}_params{key['paramset_idx']}_field{key['field_idx']}_motion_correction_results.pkl",
            )

            if mc_indices is not None:
                # store "indices" in params as tuple instead of the `slice` object
                params["motion"]["indices"] = mc_indices
        else:
            raise NotImplementedError(
                f"Field motion correction for {acq_software} scans with {method} is not yet supported in this table."
            )

        exec_dur = (datetime.now(timezone.utc) - execution_time).total_seconds() / 3600
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": exec_dur,
                "mc_params": params,
            }
        )

        self.File.insert(
            [
                {
                    **key,
                    "file_name": f.relative_to(output_dir.parent).as_posix(),
                    "file": f,
                }
                for f in output_dir.rglob(
                    f"{key['subject']}_session{key['session']}_params{key['paramset_idx']}_field{key['field_idx']}_motion_correction_results.pkl"
                )
                if f.is_file()
            ]
        )


@schema
class FieldSegmentation(dj.Computed):
    definition = """
    -> FieldMotionCorrection
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@imaging-processed
        """

    def make(self, key):
        execution_time = datetime.now(timezone.utc)
        processed_root_data_dir = scan.get_processed_root_data_dir()

        output_dir, params = (
            PreProcessing.Field.proj("processing_output_dir")
            * FieldMotionCorrection.proj("mc_params")
            & key
        ).fetch1("processing_output_dir", "mc_params")
        extra_dj_params = params.pop("extra_dj_params", {})

        output_dir = find_full_path(processed_root_data_dir, output_dir)
        acq_software = (scan.Scan & key).fetch1("acq_software")
        method = (imaging.ProcessingParamSet * imaging.ProcessingTask & key).fetch1(
            "processing_method"
        )

        if acq_software == "PrairieView" and method == "caiman":
            import multiprocessing
            import caiman as cm
            from element_interface.run_caiman import _save_mc
            from caiman.motion_correction import MotionCorrect
            from caiman.source_extraction.cnmf.cnmf import CNMF, load_CNMF
            from caiman.source_extraction.cnmf.params import CNMFParams

            mc_results = pickle.load(
                output_dir
                / f"{key['subject']}_session{key['session']}_params{key['paramset_idx']}_field{key['field_idx']}_motion_correction_results.pkl"
            )
            cnmf_mc_output_file = find_full_path(
                processed_root_data_dir, extra_dj_params["cnmf_mc_output_file"]
            )
            fnames = [
                str(find_full_path(processed_root_data_dir, f))
                for f in params["fnames"]
            ]

            output_dir = output_dir / "segmentation"
            output_dir.mkdir(parents=True, exist_ok=True)

            @memoized_result(
                uniqueness_dict={**params, **extra_dj_params},
                output_directory=output_dir,
            )
            def _run_segmentation():
                # very memory-intensive step, use 10% of available cores only
                n_processes = np.floor(multiprocessing.cpu_count() * 0.1)
                n_processes = int(os.getenv("CAIMAN_CNMF_N_PROCESSES", n_processes))
                _, dview, n_processes = cm.cluster.setup_cluster(
                    backend="multiprocessing",
                    n_processes=n_processes,
                    maxtasksperchild=1,
                )
                cnm = load_CNMF(cnmf_mc_output_file, n_processes=None, dview=None)
                cnm.params.set("data", {"fnames": fnames})

                mc = MotionCorrect(fnames, dview=cnm.dview, **cnm.params.motion)
                for k, v in mc_results.items():
                    setattr(mc, k, v)

                caiman_temp = os.environ.get("CAIMAN_TEMP")
                os.environ["CAIMAN_TEMP"] = str(output_dir)
                try:
                    base_name = pathlib.Path(fnames[0]).stem
                    fname_mc = (
                        mc.fname_tot_els
                        if cnm.params.motion["pw_rigid"]
                        else mc.fname_tot_rig
                    )
                    if all(fname_mc):
                        # F-order memmap exists (from motion correction)
                        logger.info("Generating C-order memmap file...")
                        border_to_0 = (
                            0 if mc.border_nan == "copy" else mc.border_to_0
                        )  # trim border against NaNs
                        fname_new = cm.mmapping.save_memmap(
                            fname_mc,
                            base_name=base_name + "_mc",
                            order="C",
                            var_name_hdf5=cnm.params.get("data", "var_name_hdf5"),
                            border_to_0=border_to_0,
                        )
                    else:
                        logger.info("Apply shifts, generating C-order memmap file...")
                        fname_new = mc.apply_shifts_movie(
                            fnames,
                            save_memmap=True,
                            save_base_name=base_name + "_mc",
                            order="C",
                        )
                        mc.mmap_file = [fname_new]

                    Yr, dims, T = cm.mmapping.load_memmap(fname_new)
                    images = np.reshape(Yr.T, [T] + list(dims), order="F")
                    cnm.mmap_file = fname_new

                    logger.info("Starting CNMF analysis...")
                    fit_cnm = cnm.fit(images, indices=(slice(None), slice(None)))

                    Cn = cm.summary_images.local_correlations(
                        images[:: max(T // 100, 1)], swap_dim=False
                    )
                    Cn[np.isnan(Cn)] = 0
                    fname_init_hdf5 = fname_new[:-5] + "_init.hdf5"
                    fit_cnm.save(fname_init_hdf5)
                    # fit_cnm.params.change_params({'p': self.params.get('preprocess', 'p')})
                    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
                    cnm2 = fit_cnm.refit(images, dview=cnm.dview)
                    cnm2.estimates.evaluate_components(
                        images, cnm2.params, dview=cnm.dview
                    )
                    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
                    cnm2.estimates.Cn = Cn
                    fname_hdf5 = cnm2.mmap_file[:-4] + "hdf5"
                    cnm2.save(fname_hdf5)
                except Exception as e:
                    dview.terminate()
                    raise e
                else:
                    cm.stop_server(dview=dview)
                    logger.info("CNMF analysis complete. Saving results.")
                    cnmf_output_file = pathlib.Path(fname_hdf5)

                    logger.info("Compute summary images...")
                    summary_images = {
                        "average_image": np.mean(images[:: max(T // 100, 1)], axis=0),
                        "max_image": np.max(images[:: max(T // 100, 1)], axis=0),
                        "correlation_image": Cn,
                    }

                    logger.info("Saving motion correction hdf5.")
                    _save_mc(
                        mc,
                        cnmf_output_file.as_posix(),
                        params["is3D"],
                        summary_images=summary_images,
                    )

                    if caiman_temp is not None:
                        os.environ["CAIMAN_TEMP"] = caiman_temp
                    else:
                        del os.environ["CAIMAN_TEMP"]

            _run_segmentation()
        else:
            raise NotImplementedError(
                f"Field Segmentation for {acq_software} scans with {method} is not yet supported in this table."
            )

        exec_dur = (datetime.now(timezone.utc) - execution_time).total_seconds() / 3600
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": exec_dur,
            }
        )
        # self.File.insert(
        #     [
        #         {
        #             **key,
        #             "file_name": f.relative_to(output_dir.parent).as_posix(),
        #             "file": f,
        #         }
        #         for f in output_dir.rglob("*")
        #         if f.is_file()
        #     ]
        # )


@schema
class PostProcessing(dj.Computed):
    definition = """
    -> PreProcessing
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # (hour) execution duration
    """

    @property
    def key_source(self):
        """
        Find PreProcessing entries that have finished processing for all fields
        """
        per_plane_proc = (
            PreProcessing.aggr(
                PreProcessing.Field.proj(),
                field_count="count(field_idx)",
                keep_all_rows=True,
            )
            * PreProcessing.aggr(
                FieldSegmentation.proj(),
                finished_field_count="count(field_idx)",
                keep_all_rows=True,
            )
            & "field_count = finished_field_count"
        )
        return PreProcessing & per_plane_proc

    def make(self, key):
        execution_time = datetime.now(timezone.utc)
        output_dir, method, params = (
            imaging.ProcessingTask * imaging.ProcessingParamSet & key
        ).fetch1("processing_output_dir", "processing_method", "params")

        package_version = ""
        if method == "caiman":
            import caiman

            package_version = f"CaImAn v{caiman.__version__}"

        output_dir = find_full_path(scan.get_processed_root_data_dir(), output_dir)

        exec_dur = (datetime.now(timezone.utc) - execution_time).total_seconds() / 3600
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": exec_dur,
            }
        )
        imaging.Processing.insert1(
            {
                **key,
                "processing_time": datetime.now(timezone.utc),
                "package_version": package_version,
            },
            allow_direct_insert=True,
        )
        imaging.Processing.File.insert(
            [
                {
                    **key,
                    "file_name": f.relative_to(output_dir).as_posix(),
                    "file": f,
                }
                for f in output_dir.rglob("*")
                if f.is_file()
            ],
            allow_direct_insert=True,
        )
