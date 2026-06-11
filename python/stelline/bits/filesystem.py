from typing import Any, Tuple

from holoscan.core import Application, Resource

from ..operators import (
    DummyWriterOp,
    Fbh5WriterRdmaOp,
    SimpleWriterOp,
    SimpleWriterRdmaOp,
    Uvh5WriterRdmaOp,
    Fbh5ReaderOp,
    Uvh5ReaderOp,
)
from ..registry import register_bit
from ..utils import logger


@register_bit("filesystem_bit")
def FilesystemBit(
    app: Application,
    pool: Resource,
    id: int,
    config: str,
) -> Tuple[Any, Any]:
    cfg = app.kwargs(config)
    mode = cfg.get("mode")
    file_path = cfg.get("file_path") or "./file.bin"
    logger.info("Filesystem Configuration:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  File Path: {file_path}")

    if mode == "simple_writer":
        logger.info("Creating Simple Writer operator.")
        writer_name = f"filesystem-simple-writer-{id}"
        writer_op = SimpleWriterOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "simple_writer_rdma":
        logger.info("Creating Simple Writer RDMA operator.")
        writer_name = f"filesystem-simple-writer-rdma-{id}"
        writer_op = SimpleWriterRdmaOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "dummy_writer":
        logger.info("Creating Dummy Writer operator.")
        writer_name = f"filesystem-dummy-writer-{id}"
        writer_op = DummyWriterOp(
            fragment=app,
            name=writer_name,
        )
    elif mode == "fbh5_writer_rdma":
        logger.info("Creating FBH5 Writer RDMA operator.")
        writer_name = f"filesystem-fbh5-writer-rdma-{id}"
        writer_op = Fbh5WriterRdmaOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "uvh5_writer_rdma":
        dsp_channelization_rate = cfg.get("dsp_channelization_rate")
        dsp_integration_rate = cfg.get("dsp_integration_rate")

        logger.info("  Mode Configuration:")
        logger.info(f"    DSP Channelization Rate: {dsp_channelization_rate}")
        logger.info(f"    DSP Integration Rate: {dsp_integration_rate}")

        logger.info("Creating UVH5 Writer RDMA operator.")
        writer_name = f"filesystem-uvh5-writer-rdma-{id}"
        writer_op = Uvh5WriterRdmaOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
            dsp_channelization_rate=dsp_channelization_rate,
            dsp_integration_rate=dsp_integration_rate
        )
    elif mode == "fbh5_reader":
        chunk_size = cfg.get("chunk_size") or 8192
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info("Creating FBH5 Reader operator.")
        op_name = f"filesystem-fbh5-reader-{id}"
        op = Fbh5ReaderOp(
            fragment=app,
            name=op_name,
            file_path=file_path,
            chunk_size=chunk_size,
        )
        return (op, op)
    elif mode == "uvh5_reader":
        chunk_size = cfg.get("chunk_size") or 1
        logger.info(f"  Chunk Size: {chunk_size}")
        logger.info("Creating UVH5 Reader operator.")
        op_name = f"filesystem-uvh5-reader-{id}"
        op = Uvh5ReaderOp(
            fragment=app,
            name=op_name,
            file_path=file_path,
            chunk_size=chunk_size,
        )
        return (op, op)
    else:
        raise ValueError(f"Unsupported filesystem mode: {mode}")

    return (writer_op, writer_op)


__all__ = ["FilesystemBit"]
