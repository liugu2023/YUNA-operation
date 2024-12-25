# -*- Makefile -*-

arch = Linux_MPI
setup_file = setup/Make.$(arch)

include $(setup_file)


HPCG_DEPS = src/CG.o \
	    src/CG_ref.o \
	    src/TestCG.o \
	    src/ComputeSPMV_opt.o \
	    src/ComputeResidual.o \
	    src/ExchangeHalo.o \
	    src/GenerateGeometry.o \
	    src/GenerateProblem.o \
	    src/GenerateProblem_ref.o \
	    src/CheckProblem.o \
	    src/MixedBaseCounter.o \
	    src/OptimizeProblem.o \
	    src/ReadHpcgDat.o \
	    src/ReportResults.o \
	    src/SetupHalo.o \
	    src/SetupHalo_ref.o \
	    src/TestSymmetry.o \
	    src/TestNorms.o \
	    src/WriteProblem.o \
	    src/YAML_Doc.o \
	    src/YAML_Element.o \
	    src/ComputeDotProduct.o \
	    src/ComputeDotProduct_ref.o \
	    src/mytimer.o \
	    src/ComputeOptimalShapeXYZ.o \
	    src/ComputeSPMV.o \
	    src/ComputeSPMV_ref.o \
	    src/ComputeSYMGS.o \
	    src/ComputeSYMGS_ref.o \
	    src/ComputeWAXPBY.o \
	    src/ComputeWAXPBY_ref.o \
	    src/ComputeMG_ref.o \
	    src/ComputeMG.o \
	    src/ComputeProlongation_ref.o \
	    src/ComputeRestriction_ref.o \
	    src/CheckAspectRatio.o \
	    src/OutputFile.o \
	    src/GenerateCoarseProblem.o \
	    src/init.o \
	    src/finalize.o

# These header files are included in many source files, so we recompile every file if one or more of these header is modified.
PRIMARY_HEADERS = /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/Geometry.hpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/SparseMatrix.hpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/Vector.hpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CGData.hpp \
                  /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/MGData.hpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/hpcg.hpp

all: bin/xhpcg

bin/xhpcg: src/main.o $(HPCG_DEPS)
	$(LINKER) $(LINKFLAGS) src/main.o $(HPCG_DEPS) $(HPCG_LIBS) -o bin/xhpcg

clean:
	rm -f src/*.o bin/xhpcg

.PHONY: all clean

src/main.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/main.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/CG.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CG.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/CG_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CG_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/TestCG.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestCG.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestCG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeResidual.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeResidual.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeResidual.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ExchangeHalo.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ExchangeHalo.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ExchangeHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/GenerateGeometry.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateGeometry.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateGeometry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/GenerateProblem.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateProblem.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/GenerateProblem_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateProblem_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateProblem_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/CheckProblem.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CheckProblem.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CheckProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/MixedBaseCounter.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/MixedBaseCounter.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/MixedBaseCounter.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/OptimizeProblem.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/OptimizeProblem.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/OptimizeProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ReadHpcgDat.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ReadHpcgDat.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ReadHpcgDat.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ReportResults.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ReportResults.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ReportResults.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/SetupHalo.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/SetupHalo.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/SetupHalo.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/SetupHalo_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/SetupHalo_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/SetupHalo_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/TestSymmetry.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestSymmetry.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestSymmetry.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/TestNorms.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestNorms.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/TestNorms.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/WriteProblem.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/WriteProblem.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/WriteProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/YAML_Doc.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/YAML_Doc.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/YAML_Doc.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/YAML_Element.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/YAML_Element.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/YAML_Element.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeDotProduct.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeDotProduct.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeDotProduct.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeDotProduct_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeDotProduct_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeDotProduct_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/finalize.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/finalize.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/init.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/init.cpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/mytimer.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/mytimer.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/mytimer.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeOptimalShapeXYZ.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeOptimalShapeXYZ.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeOptimalShapeXYZ.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeSPMV.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeSPMV_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeSYMGS.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSYMGS.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSYMGS.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeSYMGS_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSYMGS_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSYMGS_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeWAXPBY.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeWAXPBY.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeWAXPBY.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeWAXPBY_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeWAXPBY_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeWAXPBY_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeMG_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeMG_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeMG_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeMG.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeMG.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeMG.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeProlongation_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeProlongation_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeProlongation_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeRestriction_ref.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeRestriction_ref.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeRestriction_ref.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/GenerateCoarseProblem.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateCoarseProblem.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/GenerateCoarseProblem.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/CheckAspectRatio.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CheckAspectRatio.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/CheckAspectRatio.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/OutputFile.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/OutputFile.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/OutputFile.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

src/ComputeSPMV_opt.o: /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV_opt.cpp /home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src/ComputeSPMV_opt.hpp $(PRIMARY_HEADERS)
	$(CXX) -c $(CXXFLAGS) -I/home/liugu/Operations-Department-Operations/optimize/test1/hpcg/src $< -o $@

# 添加优化编译选项
CXXFLAGS += -O3 -fopenmp -march=native -ffast-math

