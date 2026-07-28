FROM rexgroundingct-voxtell:cu126

USER root

COPY external/TotalSegmentator /opt/TotalSegmentator

RUN python -m pip install \
      --no-deps \
      --no-build-isolation \
      /opt/TotalSegmentator \
    && python -m pip install \
      --no-deps \
      dicom2nifti==2.6.2 \
      pydicom==3.0.2 \
      python-gdcm==3.2.6 \
    && python -c "import importlib.metadata as m; assert m.version('TotalSegmentator') == '2.16.0'"

LABEL org.opencontainers.image.title="ReXGroundingCT TotalSegmentator runtime" \
      org.opencontainers.image.version="2.16.0-cu126" \
      org.opencontainers.image.source="https://github.com/wasserth/TotalSegmentator" \
      org.opencontainers.image.revision="099b679c76350effc701889d22f25fdbf060cea6"
