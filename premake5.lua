include "libs/PrLib"

workspace "PathCuts"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

externalproject "prlib"
	location "libs/PrLib/build" 
    kind "StaticLib"
    language "C++"

function enableOrochi()
    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }
    
    -- Cuda
    defines {"OROCHI_ENABLE_CUEW"}
    includedirs {"$(CUDA_PATH)/include"}
end

project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main.cpp" }
    files { "*.h" }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    enableOrochi()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Main_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Main")
        optimize "Full"
    filter{}

project "main_gpu"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main_gpu.cpp" }
    files { "*.h" }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    enableOrochi()

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("main_gpu_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("main_gpu")
        optimize "Full"
    filter{}
    
project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++14"
    buildoptions "/bigobj"

    -- Src
    files { "unittest.cpp", "catch_amalgamated.cpp", "catch_amalgamated.hpp" }
    includedirs { "." }

    enableOrochi()

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Unittest_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Unittest")
        optimize "Full"
    filter{}

project "copying"    
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin",
        "{COPYFILE} ../*.cu ../bin/kernels",
        "{COPYFILE} ../*.h ../bin/kernels",
    }