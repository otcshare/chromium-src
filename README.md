System requirements and environment  setup can follow [the instructions of building chromium on windows](https://chromium.googlesource.com/chromium/src/+/main/docs/windows_build_instructions.md)

## Get the code
* Create a chromium directory for the checkout and change to it (you can call this whatever you like and put it wherever you like, as long as the full path has no spaces):  
`$ mkdir chromium && cd chromium`

* Clone the chromium-src repository:  
`$ git clone https://github.com/otcshare/chromium-src.git`

* Create .gclient file and edit the file to contain the following arguments:  
`solutions = [`  
`  {`  
`    "url": "http://github.com/otcshare/chromium-src.git@webnn_mojo",`  
`    "managed": False,`  
`    "name": "chromium-src",`  
`    "deps_file": ".DEPS.git",`  
`    "custom_deps": {},`  
`  },`  
`]`


## Update your checkout and run the hooks
* The remaining instructions assume you have switched to the src directory:  
`$ cd chromium-src`

* To update an existing checkout, you can run:  
`$ gclient sync`

## Setting up the build
* Chromium uses Ninja as its main build tool along with a tool called GN to generate .ninja files. You can create any number of build directories with different configurations. To create a build directory, run:  
`$ gn gen out/Default`

* To config the flags for WebNN, run gn args out/Default and edit the file to contain the following arguments:  
`$ gn args out/Default`  
`target_os = "win"`  
`target_cpu = "x64"`  
`is_debug = false`  
`is_component_build = false`
  > You only have to run this once for each new build directory, Ninja will update the build files as needed.  
    You can replace Default with another name, but it should be a subdirectory of out.

## Build Chromium
* Build Chromium (the “chrome” target) with Ninja using the command:  
`$ autoninja -C out\Default chrome`

## Run Chromium
* Once it is built, you can simply run the browser:  
`$ out\Default\chrome.exe  --enable-blink-test-features`

* Search via "chrome://flags/" URL in browser to enable flags for WebNN, you may need restart browser after enabling flags:  
`Experimental Web Platform features`  
`Enables Machine Learning Neural Network Web Platform API`

## Test
* Setup and Run Webnn-samples 
```sh
> git clone --recurse-submodules https://github.com/fujunwei/webnn-samples.git
> git fetch origin webnn_mojo
> cd webnn-samples & npm install
> npm start
```
Navigate the open browser to http://localhost:8080, select image classification sample and run MobileNet V2(NCHW) for GPU that is currently supported.
