import AppKit
import AVFoundation
import Metal
import MetalKit
import CoreVideo
import UniformTypeIdentifiers

private let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float yaw;
    float pitch;
    float fovRadians;
    float aspect;
    float flipX;
    float flipY;
    float2 padding;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertexMain(uint vertexID [[vertex_id]]) {
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };

    float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 fragmentMain(
    VertexOut in [[stage_in]],
    texture2d<float> lumaTexture [[texture(0)]],
    texture2d<float> chromaTexture [[texture(1)]],
    sampler videoSampler [[sampler(0)]],
    constant Uniforms &uniforms [[buffer(0)]]
) {
    if (is_null_texture(lumaTexture) || is_null_texture(chromaTexture)) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float2 uv = in.uv;

    if (uniforms.flipX > 0.5) {
        uv.x = 1.0 - uv.x;
    }

    if (uniforms.flipY > 0.5) {
        uv.y = 1.0 - uv.y;
    }

    float ndcX = uv.x * 2.0 - 1.0;
    float ndcY = 1.0 - uv.y * 2.0;

    float tanY = tan(uniforms.fovRadians * 0.5);
    float tanX = tanY * uniforms.aspect;

    float3 cameraDirection = normalize(float3(
        ndcX * tanX,
        ndcY * tanY,
        -1.0
    ));

    float cy = cos(uniforms.yaw);
    float sy = sin(uniforms.yaw);
    float cp = cos(uniforms.pitch);
    float sp = sin(uniforms.pitch);

    float3 pitched = float3(
        cameraDirection.x,
        cameraDirection.y * cp - cameraDirection.z * sp,
        cameraDirection.y * sp + cameraDirection.z * cp
    );

    float3 direction = float3(
        pitched.x * cy + pitched.z * sy,
        pitched.y,
        -pitched.x * sy + pitched.z * cy
    );

    float longitude = atan2(direction.x, -direction.z);
    float latitude = asin(clamp(direction.y, -1.0, 1.0));

    float2 equirectUV = float2(
        0.5 + longitude / (2.0 * 3.14159265),
        0.5 - latitude / 3.14159265
    );

    float luma = lumaTexture.sample(videoSampler, equirectUV).r;
    float2 chroma = chromaTexture.sample(videoSampler, equirectUV).rg;

    // BT.709 limited-range YCbCr to RGB.
    float y = (luma - 0.062745) * 1.164383;
    float cb = chroma.x - 0.5;
    float cr = chroma.y - 0.5;

    float r = y + 1.792741 * cr;
    float g = y - 0.213249 * cb - 0.532909 * cr;
    float b = y + 2.112402 * cb;

    return float4(clamp(r, 0.0, 1.0),
                  clamp(g, 0.0, 1.0),
                  clamp(b, 0.0, 1.0),
                  1.0);
}
"""

struct Uniforms {
    var yaw: Float = 0
    var pitch: Float = 0
    var fovRadians: Float = 90 * Float.pi / 180
    var aspect: Float = 16.0 / 9.0
    var flipX: Float = 0
    var flipY: Float = 0
    var padding: SIMD2<Float> = .zero
}

final class VRMetalView: MTKView {
    weak var controller: ViewController?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.acceptsMouseMovedEvents = true
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()

        for area in trackingAreas {
            removeTrackingArea(area)
        }

        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .activeInKeyWindow],
            owner: self
        )

        addTrackingArea(area)
    }

    override func mouseDown(with event: NSEvent) {
        controller?.showControls()

        if event.clickCount == 2 {
            window?.toggleFullScreen(nil)
            return
        }

        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        controller?.showControls()
        controller?.rotate(dx: event.deltaX, dy: event.deltaY)
    }

    override func mouseMoved(with event: NSEvent) {
        controller?.showControls()
    }

    override func scrollWheel(with event: NSEvent) {
        controller?.showControls()

        let deltaX = event.scrollingDeltaX
        let deltaY = event.scrollingDeltaY

        // Horizontal scrolling seeks the video.
        if abs(deltaX) > abs(deltaY) {
            // Natural scrolling: swipe right = seek forward.
            controller?.seek(by: deltaX * 0.1)
            return
        }

        // Vertical scrolling zooms.
        controller?.zoom(by: -deltaY * 0.15)
    }

    override func magnify(with event: NSEvent) {
        controller?.showControls()
        controller?.zoom(by: event.magnification * 25)
    }

    override func keyDown(with event: NSEvent) {
        guard let controller = controller else {
            super.keyDown(with: event)
            return
        }

        switch event.keyCode {
        case 49: controller.togglePlayback()
        case 123: controller.seek(by: -10)
        case 124: controller.seek(by: 10)
        case 126: controller.zoom(by: -5)
        case 125: controller.zoom(by: 5)
        case 3: controller.toggleFlip()
        case 46: controller.toggleMirror()
        case 2: controller.toggleDebugHUD()
        default:
            super.keyDown(with: event)
        }
    }

    override var acceptsFirstResponder: Bool { true }
}

final class ViewController: NSObject {
    let metalView: VRMetalView
    let player = AVPlayer()

    private let device = MTLCreateSystemDefaultDevice()!
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private let pipelineState: MTLRenderPipelineState
    private let samplerState: MTLSamplerState?

    private var videoOutput: AVPlayerItemVideoOutput?
    private var textureCache: CVMetalTextureCache?

    private var persistentLumaTexture: MTLTexture?
    private var persistentChromaTexture: MTLTexture?
    private var persistentTextureWidth = 0
    private var persistentTextureHeight = 0

    private var uniforms = Uniforms()
    private var timeObserver: Any?
    private var statusObserver: NSKeyValueObservation?
    private var itemStatusObserver: NSKeyValueObservation?
    private var isUserSeeking = false
    private var controlsHideTimer: Timer?

    private let controlsContainer = NSVisualEffectView()
    private let playButton = NSButton(title: "Pause", target: nil, action: nil)
    private let openButton = NSButton(title: "Open", target: nil, action: nil)
    private let timeLabel = NSTextField(labelWithString: "00:00 / 00:00")
    private let seekSlider = NSSlider(value: 0, minValue: 0, maxValue: 1, target: nil, action: nil)
    private let debugLabel = NSTextField(labelWithString: "")

    private var debugEnabled = false

    private var renderFrameCount = 0
    private var textureFrameCount = 0
    private var stallCount = 0
    private var outputRecoveryCount = 0
    private var consecutiveNoNewFrameCount = 0
    private var lastStatsUpdateTime: Double = 0
    private var lastTextureUpdateTime: Double = 0
    private var lastTextureIntervalMs: Double = 0
    private var cpuFrameTimeMs: Double = 0
    private var gpuFrameTimeMs: Double = 0
    private var lastOutputRecoveryTime: Double?
    private var lastPresentTime: Double = 0

    private var lastPlayerTime: CMTime = .invalid
    private var lastRequestedItemTime: CMTime = .invalid
    private var lastCopiedItemTime: CMTime = .invalid
    private var lastTextureWidth = 0
    private var lastTextureHeight = 0

    init(frame: NSRect) {
        commandQueue = device.makeCommandQueue()!

        do {
            library = try device.makeLibrary(source: metalShaderSource, options: nil)
        } catch {
            fatalError("Unable to compile Metal shader: \(error)")
        }

        let vertexFunction = library.makeFunction(name: "vertexMain")
        let fragmentFunction = library.makeFunction(name: "fragmentMain")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Unable to create Metal render pipeline state: \(error)")
        }

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .notMipmapped
        samplerDescriptor.sAddressMode = .repeat
        samplerDescriptor.tAddressMode = .clampToEdge
        samplerState = device.makeSamplerState(descriptor: samplerDescriptor)

        metalView = VRMetalView(frame: frame, device: device)

        super.init()

        metalView.controller = self
        metalView.delegate = self
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        metalView.framebufferOnly = true

        metalView.isPaused = false
        metalView.enableSetNeedsDisplay = false
        metalView.preferredFramesPerSecond = 60

        metalView.autoresizingMask = [.width, .height]
        metalView.translatesAutoresizingMaskIntoConstraints = false

        var cache: CVMetalTextureCache?
        let cacheResult = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            device,
            nil,
            &cache
        )

        if cacheResult != kCVReturnSuccess {
            fatalError("Unable to create CoreVideo Metal texture cache: \(cacheResult)")
        }

        textureCache = cache

        player.automaticallyWaitsToMinimizeStalling = false

        setupControls(in: frame)
        addTimeObserver()
        observePlaybackStalls()
    }

    private func setupControls(in frame: NSRect) {
        let container = NSView(frame: frame)
        container.autoresizingMask = [.width, .height]
        container.wantsLayer = true

        container.addSubview(metalView)

        controlsContainer.material = .hudWindow
        controlsContainer.blendingMode = .behindWindow
        controlsContainer.state = .active
        controlsContainer.translatesAutoresizingMaskIntoConstraints = false

        playButton.bezelStyle = .rounded
        playButton.controlSize = .regular
        playButton.target = self
        playButton.action = #selector(togglePlayback)
        playButton.translatesAutoresizingMaskIntoConstraints = false

        openButton.bezelStyle = .rounded
        openButton.controlSize = .regular
        openButton.target = self
        openButton.action = #selector(openFilePicker)
        openButton.translatesAutoresizingMaskIntoConstraints = false

        seekSlider.minValue = 0
        seekSlider.maxValue = 1
        seekSlider.doubleValue = 0
        seekSlider.target = self
        seekSlider.action = #selector(seekSliderChanged(_:))
        seekSlider.translatesAutoresizingMaskIntoConstraints = false

        timeLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        timeLabel.alignment = .center
        timeLabel.translatesAutoresizingMaskIntoConstraints = false

        debugLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular)
        debugLabel.textColor = .white
        debugLabel.drawsBackground = false
        debugLabel.isBezeled = false
        debugLabel.isEditable = false
        debugLabel.translatesAutoresizingMaskIntoConstraints = false
        debugLabel.stringValue = "Debug HUD enabled — press D to hide"
        debugLabel.isHidden = true

        container.addSubview(debugLabel)
        container.addSubview(controlsContainer)
        controlsContainer.addSubview(playButton)
        controlsContainer.addSubview(openButton)
        controlsContainer.addSubview(seekSlider)
        controlsContainer.addSubview(timeLabel)

        NSLayoutConstraint.activate([
            metalView.topAnchor.constraint(equalTo: container.topAnchor),
            metalView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            metalView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            metalView.bottomAnchor.constraint(equalTo: container.bottomAnchor),

            debugLabel.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            debugLabel.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 12),
            debugLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 420),

            controlsContainer.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            controlsContainer.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            controlsContainer.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            controlsContainer.heightAnchor.constraint(equalToConstant: 52),

            playButton.leadingAnchor.constraint(equalTo: controlsContainer.leadingAnchor, constant: 14),
            playButton.centerYAnchor.constraint(equalTo: controlsContainer.centerYAnchor),
            playButton.widthAnchor.constraint(equalToConstant: 88),

            openButton.leadingAnchor.constraint(equalTo: playButton.trailingAnchor, constant: 10),
            openButton.centerYAnchor.constraint(equalTo: controlsContainer.centerYAnchor),
            openButton.widthAnchor.constraint(equalToConstant: 78),

            seekSlider.leadingAnchor.constraint(equalTo: openButton.trailingAnchor, constant: 14),
            seekSlider.trailingAnchor.constraint(equalTo: timeLabel.leadingAnchor, constant: -14),
            seekSlider.centerYAnchor.constraint(equalTo: controlsContainer.centerYAnchor),

            timeLabel.trailingAnchor.constraint(equalTo: controlsContainer.trailingAnchor, constant: -14),
            timeLabel.centerYAnchor.constraint(equalTo: controlsContainer.centerYAnchor),
            timeLabel.widthAnchor.constraint(equalToConstant: 118)
        ])

        metalView.window?.contentView = container
    }

    private func addTimeObserver() {
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.5, preferredTimescale: 600),
            queue: .main
        ) { [weak self] (time: CMTime) in
            guard let self = self else { return }
            self.updatePlaybackUI(currentTime: time)
        }
    }

    private func observePlaybackStalls() {
        statusObserver = player.observe(
            \.timeControlStatus,
            options: [.new, .old]
        ) { [weak self] player, change in
            guard let self = self else { return }

            let oldStatus = change.oldValue ?? .paused
            let newStatus = change.newValue ?? .paused

            if oldStatus != .waitingToPlayAtSpecifiedRate,
               newStatus == .waitingToPlayAtSpecifiedRate {
                self.stallCount += 1
            }

            if newStatus == .paused {
                self.showControls()
            }
        }
    }

    func load(url: URL) {
        let item = AVPlayerItem(url: url)
        item.preferredForwardBufferDuration = 5

        player.replaceCurrentItem(with: item)
        player.play()

        observeItemReady(item)

        persistentLumaTexture = nil
        persistentChromaTexture = nil
        persistentTextureWidth = 0
        persistentTextureHeight = 0

        lastTextureUpdateTime = 0
        lastPlayerTime = .invalid
        lastRequestedItemTime = .invalid
        lastCopiedItemTime = .invalid
        lastTextureWidth = 0
        lastTextureHeight = 0
        consecutiveNoNewFrameCount = 0
        outputRecoveryCount = 0
        lastOutputRecoveryTime = nil

        seekSlider.doubleValue = 0
        timeLabel.stringValue = "00:00 / 00:00"
        playButton.title = "Pause"

        showControls()
    }

    private func observeItemReady(_ item: AVPlayerItem) {
        itemStatusObserver = item.observe(
            \.status,
            options: [.initial, .new]
        ) { [weak self] item, _ in
            guard item.status == .readyToPlay else {
                return
            }

            DispatchQueue.main.async {
                guard let self = self,
                      self.player.currentItem === item else {
                    return
                }

                if let oldOutput = self.videoOutput {
                    item.remove(oldOutput)
                }

                let output = self.makeVideoOutput()
                item.add(output)
                self.videoOutput = output
                self.consecutiveNoNewFrameCount = 0
            }
        }
    }

    private func makeVideoOutput() -> AVPlayerItemVideoOutput {
        let output = AVPlayerItemVideoOutput(
            pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: [
                    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                ],
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
        )

        output.suppressesPlayerRendering = true
        return output
    }

    private func recreateVideoOutputIfNeeded() {
        guard player.timeControlStatus == .playing,
              consecutiveNoNewFrameCount >= 120,
              let item = player.currentItem,
              let oldOutput = videoOutput else {
            return
        }

        let now = CACurrentMediaTime()

        if let lastRecoveryTime = lastOutputRecoveryTime,
           now - lastRecoveryTime < 5.0 {
            return
        }

        lastOutputRecoveryTime = now
        outputRecoveryCount += 1
        consecutiveNoNewFrameCount = 0

        DispatchQueue.main.async { [weak self] in
            guard let self = self,
                  self.player.currentItem === item else {
                return
            }

            item.remove(oldOutput)

            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0 / 60.0) { [weak self] in
                guard let self = self,
                      self.player.currentItem === item else {
                    return
                }

                let newOutput = self.makeVideoOutput()
                item.add(newOutput)
                self.videoOutput = newOutput
            }
        }
    }

    private func updateTexture() -> (
        luma: CVMetalTexture,
        chroma: CVMetalTexture,
        width: Int,
        height: Int,
        requestedTime: CMTime
    )? {
        guard let output = videoOutput,
              let textureCache = textureCache else {
            return nil
        }

        let requestedTime = output.itemTime(forHostTime: CACurrentMediaTime())
        let playerTime = player.currentTime()

        lastPlayerTime = playerTime
        lastRequestedItemTime = requestedTime

        let hasNewBuffer = output.hasNewPixelBuffer(forItemTime: requestedTime)

        guard requestedTime.isValid,
              requestedTime.seconds.isFinite,
              hasNewBuffer,
              let pixelBuffer = output.copyPixelBuffer(
                  forItemTime: requestedTime,
                  itemTimeForDisplay: nil
              ) else {
            consecutiveNoNewFrameCount += 1
            recreateVideoOutputIfNeeded()
            return nil
        }

        consecutiveNoNewFrameCount = 0
        lastCopiedItemTime = requestedTime

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var lumaTexture: CVMetalTexture?
        let lumaResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .r8Unorm,
            width,
            height,
            0,
            &lumaTexture
        )

        var chromaTexture: CVMetalTexture?
        let chromaResult = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            .rg8Unorm,
            width / 2,
            height / 2,
            1,
            &chromaTexture
        )

        guard lumaResult == kCVReturnSuccess,
              chromaResult == kCVReturnSuccess,
              let luma = lumaTexture,
              let chroma = chromaTexture else {
            return nil
        }

        lastTextureWidth = width
        lastTextureHeight = height
        textureFrameCount += 1

        let now = CACurrentMediaTime()

        if lastTextureUpdateTime > 0 {
            lastTextureIntervalMs = (now - lastTextureUpdateTime) * 1000
        }

        lastTextureUpdateTime = now

        return (luma, chroma, width, height, requestedTime)
    }

    private func ensurePersistentTextures(width: Int, height: Int) {
        guard persistentLumaTexture == nil ||
              persistentChromaTexture == nil ||
              persistentTextureWidth != width ||
              persistentTextureHeight != height else {
            return
        }

        let lumaDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        lumaDescriptor.usage = [.shaderRead, .renderTarget]
        lumaDescriptor.storageMode = .private

        let chromaDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg8Unorm,
            width: width / 2,
            height: height / 2,
            mipmapped: false
        )
        chromaDescriptor.usage = [.shaderRead, .renderTarget]
        chromaDescriptor.storageMode = .private

        persistentLumaTexture = device.makeTexture(descriptor: lumaDescriptor)
        persistentChromaTexture = device.makeTexture(descriptor: chromaDescriptor)
        persistentTextureWidth = width
        persistentTextureHeight = height
    }

    private func updatePlaybackUI(currentTime: CMTime) {
        guard let item = player.currentItem else { return }

        let duration = item.duration.seconds
        guard duration.isFinite, duration > 0 else { return }

        let current = currentTime.seconds

        if !isUserSeeking {
            seekSlider.doubleValue = current / duration
        }

        timeLabel.stringValue = "\(format(current)) / \(format(duration))"
        playButton.title = player.timeControlStatus == .playing ? "Pause" : "Play"
    }

    private func format(_ seconds: Double) -> String {
        let total = Int(max(0, seconds.rounded()))
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    func showControls() {
        controlsContainer.isHidden = false
        controlsHideTimer?.invalidate()

        guard player.timeControlStatus == .playing else {
            return
        }

        controlsHideTimer = Timer.scheduledTimer(
            withTimeInterval: 3.0,
            repeats: false
        ) { [weak self] _ in
            guard let self = self,
                  self.player.timeControlStatus == .playing else {
                return
            }

            self.controlsContainer.isHidden = true
        }
    }

    func rotate(dx: CGFloat, dy: CGFloat) {
        let sensitivity: Float = 0.004

        uniforms.yaw -= Float(dx) * sensitivity

        var pitch = uniforms.pitch - Float(dy) * sensitivity
        pitch = max(-Float.pi / 2 + 0.01, min(Float.pi / 2 - 0.01, pitch))
        uniforms.pitch = pitch
    }

    func zoom(by amount: CGFloat) {
        var fovDegrees = uniforms.fovRadians * 180 / Float.pi
        fovDegrees = max(25, min(110, fovDegrees + Float(amount)))
        uniforms.fovRadians = fovDegrees * Float.pi / 180
    }

    @objc func togglePlayback() {
        if player.timeControlStatus == .playing {
            player.pause()
            playButton.title = "Play"
            showControls()
        } else {
            player.play()
            playButton.title = "Pause"
            showControls()
        }
    }

    @objc func openFilePicker() {
        let panel = NSOpenPanel()
        panel.title = "Choose a 360° Video"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]

        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    func seek(by seconds: Double) {
        guard let item = player.currentItem else { return }

        let duration = item.duration.seconds
        guard duration.isFinite, duration > 0 else { return }

        let target = max(0, min(duration - 0.05, player.currentTime().seconds + seconds))
        seek(to: target)
    }

    @objc private func seekSliderChanged(_ sender: NSSlider) {
        guard let item = player.currentItem else { return }

        let duration = item.duration.seconds
        guard duration.isFinite, duration > 0 else { return }

        isUserSeeking = true
        seek(to: sender.doubleValue * duration)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { [weak self] in
            self?.isUserSeeking = false
        }
    }

    private func seek(to seconds: Double) {
        player.seek(
            to: CMTime(seconds: seconds, preferredTimescale: 600),
            toleranceBefore: .zero,
            toleranceAfter: .zero
        )
    }

    func toggleFlip() {
        uniforms.flipY = uniforms.flipY > 0.5 ? 0 : 1
    }

    func toggleMirror() {
        uniforms.flipX = uniforms.flipX > 0.5 ? 0 : 1
    }

    func toggleDebugHUD() {
        debugEnabled.toggle()
        debugLabel.isHidden = !debugEnabled
    }

    private func updateDebugHUD() {
        guard debugEnabled else { return }

        let now = CACurrentMediaTime()

        guard lastStatsUpdateTime > 0 else {
            lastStatsUpdateTime = now
            return
        }

        let elapsed = now - lastStatsUpdateTime

        guard elapsed >= 0.5 else {
            return
        }

        let renderFPS = Double(renderFrameCount) / elapsed
        let textureFPS = Double(textureFrameCount) / elapsed

        renderFrameCount = 0
        textureFrameCount = 0
        lastStatsUpdateTime = now

        let status: String
        switch player.timeControlStatus {
        case .playing:
            status = "playing"
        case .paused:
            status = "paused"
        case .waitingToPlayAtSpecifiedRate:
            status = "stalled/waiting"
        @unknown default:
            status = "unknown"
        }

        var bufferedAhead = 0.0

        if let item = player.currentItem,
           let range = item.loadedTimeRanges.first?.timeRangeValue {
            bufferedAhead = max(0, range.end.seconds - player.currentTime().seconds)
        }

        let textureAgeMs = lastTextureUpdateTime > 0
            ? (now - lastTextureUpdateTime) * 1000
            : 0

        let outputLagMs = lastCopiedItemTime.isValid
            ? (lastCopiedItemTime.seconds - player.currentTime().seconds) * 1000
            : 0

        let textureSize = lastTextureWidth > 0 && lastTextureHeight > 0
            ? "\(lastTextureWidth)x\(lastTextureHeight)"
            : "none"

        let text = """
        Render: \(String(format: "%.1f", renderFPS)) FPS
        New video frames: \(String(format: "%.1f", textureFPS)) FPS
        Missed-frame streak: \(consecutiveNoNewFrameCount)
        Frame interval: \(String(format: "%.1f", lastTextureIntervalMs)) ms
        CPU frame: \(String(format: "%.2f", cpuFrameTimeMs)) ms
        GPU frame: \(String(format: "%.2f", gpuFrameTimeMs)) ms
        Texture: \(textureSize), age \(String(format: "%.1f", textureAgeMs)) ms
        Output lag: \(String(format: "%.1f", outputLagMs)) ms
        Player: \(status), stalls: \(stallCount)
        Output recoveries: \(outputRecoveryCount)
        Buffered ahead: \(String(format: "%.2f", bufferedAhead)) s
        """

        debugLabel.stringValue = text
    }
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        uniforms.aspect = Float(size.width / max(1, size.height))
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()

        // Lock presentation to 60 FPS, even on 120 Hz displays.
        if now - lastPresentTime < (1.0 / 60.0) - 0.0005 {
            return
        }

        lastPresentTime = now

        let cpuStart = CACurrentMediaTime()

        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            cpuFrameTimeMs = (CACurrentMediaTime() - cpuStart) * 1000
            return
        }

        let gpuStart = CACurrentMediaTime()

        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }

            let gpuDurationMs = (CACurrentMediaTime() - gpuStart) * 1000

            DispatchQueue.main.async {
                self.gpuFrameTimeMs = gpuDurationMs
            }
        }

        var temporaryLumaTexture: CVMetalTexture?
        var temporaryChromaTexture: CVMetalTexture?

        if let frame = updateTexture() {
            temporaryLumaTexture = frame.luma
            temporaryChromaTexture = frame.chroma

            ensurePersistentTextures(width: frame.width, height: frame.height)

            if let luma = persistentLumaTexture,
               let chroma = persistentChromaTexture,
               let sourceLumaTexture = CVMetalTextureGetTexture(frame.luma),
               let sourceChromaTexture = CVMetalTextureGetTexture(frame.chroma),
               let blitEncoder = commandBuffer.makeBlitCommandEncoder() {

                blitEncoder.copy(
                    from: sourceLumaTexture,
                    sourceSlice: 0,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: frame.width, height: frame.height, depth: 1),
                    to: luma,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )

                blitEncoder.copy(
                    from: sourceChromaTexture,
                    sourceSlice: 0,
                    sourceLevel: 0,
                    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                    sourceSize: MTLSize(width: frame.width / 2, height: frame.height / 2, depth: 1),
                    to: chroma,
                    destinationSlice: 0,
                    destinationLevel: 0,
                    destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
                )

                blitEncoder.endEncoding()
            }
        }

        guard let lumaTexture = persistentLumaTexture,
              let chromaTexture = persistentChromaTexture,
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        uniforms.aspect = Float(view.drawableSize.width / max(1, view.drawableSize.height))

        encoder.setRenderPipelineState(pipelineState)

        if let samplerState = samplerState {
            encoder.setFragmentSamplerState(samplerState, index: 0)
        }

        encoder.setFragmentTexture(lumaTexture, index: 0)
        encoder.setFragmentTexture(chromaTexture, index: 1)

        encoder.setFragmentBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 0
        )

        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        if let temporaryLumaTexture = temporaryLumaTexture,
           let temporaryChromaTexture = temporaryChromaTexture {
            commandBuffer.addCompletedHandler { _ in
                _ = temporaryLumaTexture
                _ = temporaryChromaTexture
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        cpuFrameTimeMs = (CACurrentMediaTime() - cpuStart) * 1000
        renderFrameCount += 1

        updateDebugHUD()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var controller: ViewController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let arguments = CommandLine.arguments

        if arguments.contains("-h") || arguments.contains("--help") {
            printHelp()
            exit(0)
        }

        NSApp.setActivationPolicy(.regular)

        let contentRect = NSRect(x: 0, y: 0, width: 1280, height: 720)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "macOS VR Video Player"
        window.center()

        controller = ViewController(frame: contentRect)
        window.contentView = controller.metalView.superview

        makeMenu()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        let videoArguments = arguments.dropFirst()

        if let first = videoArguments.first {
            controller.load(url: URL(fileURLWithPath: first))
        } else {
            controller.openFilePicker()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func printHelp() {
        let help = """
        macOS VR Video Player

        Usage:
          VRPlayer [options] [video-file]

        Options:
          -h, --help          Show this help message
          <video-file>        Open the specified video file

        Playback controls:
          Space               Play / pause
          Left arrow          Seek backward 10 seconds
          Right arrow         Seek forward 10 seconds
          Up arrow            Zoom out
          Down arrow          Zoom in
          F                   Toggle vertical flip
          M                   Toggle horizontal mirror
          D                   Toggle debug HUD

        Mouse controls:
          Drag                Look around the 360° video
          Scroll up/down      Zoom
          Scroll left/right   Seek backward / forward
          Pinch               Zoom
          Double-click        Toggle fullscreen

        Controls bar:
          Open                Choose a video file
          The playback bar automatically hides after 3 seconds during playback.
          Move the mouse to show it again.
        """

        print(help)
    }

    private func makeMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(title: "Quit VR Video Player",
                                   action: #selector(NSApplication.terminate(_:)),
                                   keyEquivalent: "q"))
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let fileMenuItem = NSMenuItem()
        let fileMenu = NSMenu(title: "File")

        let openItem = NSMenuItem(title: "Open Video…",
                                  action: #selector(openFile),
                                  keyEquivalent: "o")
        openItem.target = self
        fileMenu.addItem(openItem)

        let playPauseItem = NSMenuItem(title: "Play / Pause",
                                       action: #selector(togglePlayback),
                                       keyEquivalent: " ")
        playPauseItem.target = self
        fileMenu.addItem(playPauseItem)

        fileMenuItem.submenu = fileMenu
        mainMenu.addItem(fileMenuItem)

        NSApp.mainMenu = mainMenu
    }

    @objc func openFile() {
        controller.openFilePicker()
    }

    @objc func togglePlayback() {
        controller.togglePlayback()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
