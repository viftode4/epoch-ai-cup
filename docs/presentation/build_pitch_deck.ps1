param(
    [string]$OutputPath = "docs/presentation/ai-cup-congress-pitch-v1.pptx"
)

$ErrorActionPreference = 'Stop'

function Set-Rgb([int]$r,[int]$g,[int]$b){ return $r + ($g * 256) + ($b * 65536) }

$BG = Set-Rgb 17 20 24
$BG2 = Set-Rgb 28 33 39
$TEXT = Set-Rgb 247 241 232
$MUTED = Set-Rgb 182 188 196
$ORANGE = Set-Rgb 255 122 26
$TEAL = Set-Rgb 25 197 183
$RED = Set-Rgb 255 94 91
$GOLD = Set-Rgb 255 209 102

function Add-TextBox($slide, $left, $top, $width, $height, $text, $fontSize=28, $bold=$false, $color=$TEXT, $fontName='Segoe UI', $align=1) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    $tf = $shape.TextFrame.TextRange
    $tf.Text = $text
    $tf.Font.Name = $fontName
    $tf.Font.Size = $fontSize
    $tf.Font.Bold = [int]($bold)
    $tf.Font.Color.RGB = [int]$color
    $shape.TextFrame2.WordWrap = -1
    $shape.TextFrame.VerticalAnchor = 1
    return $shape
}

function Add-FillRect($slide, $left, $top, $width, $height, $fillColor, $lineColor=$fillColor, $radius=0) {
    $shapeType = if ($radius -gt 0) { 5 } else { 1 }
    $shape = $slide.Shapes.AddShape($shapeType, $left, $top, $width, $height)
    $shape.Fill.ForeColor.RGB = [int]$fillColor
    $shape.Line.ForeColor.RGB = [int]$lineColor
    $shape.Line.Weight = 1.25
    return $shape
}

function Add-BaseSlide($presentation, $index) {
    $slide = $presentation.Slides.Add($index, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.ForeColor.RGB = $BG
    [void](Add-FillRect $slide 0 0 1280 20 $ORANGE)
    [void](Add-FillRect $slide 0 700 1280 20 $TEAL)
    [void](Add-TextBox $slide 1080 30 160 30 'AI Cup 2026 finalist pitch' 12 $true $MUTED 'Segoe UI' 3)
    return $slide
}

function Add-CardTitle($slide, $left, $top, $width, $title, $body, $accent) {
    $card = Add-FillRect $slide $left $top $width 190 $BG2 $BG2 18
    $card.Line.ForeColor.RGB = $accent
    $card.Line.Weight = 1.5
    [void](Add-TextBox $slide ($left+20) ($top+18) ($width-40) 36 $title 22 $true $TEXT 'Segoe UI Semibold')
    [void](Add-TextBox $slide ($left+20) ($top+60) ($width-40) 110 $body 15 $false $MUTED 'Segoe UI')
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideSize = 16

# Slide 1
$slide = Add-BaseSlide $presentation 1
[void](Add-TextBox $slide 70 90 650 160 'Bird-safe wind energy needs trustworthy AI' 29 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 240 650 100 'AI Cup 2026 finalist pitch on radar classification, disciplined research workflow, and deployable decision support.' 20 $false $MUTED)
$b1 = Add-FillRect $slide 70 355 120 26 $ORANGE $ORANGE 18; $b1.Line.Visible = 0
$b2 = Add-FillRect $slide 205 355 130 26 $TEAL $TEAL 18; $b2.Line.Visible = 0
$b3 = Add-FillRect $slide 350 355 160 26 $BG2 $BG2 18; $b3.Line.Visible = 0
[void](Add-TextBox $slide 70 358 120 20 'Biodiversity' 12 $true $TEXT 'Segoe UI Semibold' 2)
[void](Add-TextBox $slide 205 358 130 20 'Wind energy' 12 $true $TEXT 'Segoe UI Semibold' 2)
[void](Add-TextBox $slide 350 358 160 20 'Deployable AI' 12 $true $TEXT 'Segoe UI Semibold' 2)
$circle1 = $slide.Shapes.AddShape(9, 800, 110, 300, 300)
$circle1.Fill.ForeColor.RGB = $BG2; $circle1.Line.Visible = 0
$circle2 = $slide.Shapes.AddShape(9, 930, 210, 190, 190)
$circle2.Fill.ForeColor.RGB = $ORANGE; $circle2.Line.Visible = 0; $circle2.Fill.Transparency = 0.18
$circle3 = $slide.Shapes.AddShape(9, 760, 320, 160, 160)
$circle3.Fill.ForeColor.RGB = $TEAL; $circle3.Line.Visible = 0; $circle3.Fill.Transparency = 0.15
[void](Add-TextBox $slide 810 165 250 120 'We are not just presenting a model.
We are presenting a credible way to build deployable AI for a difficult real-world problem.' 20 $true $TEXT 'Segoe UI Semibold' 2)

# Slide 2
$slide = Add-BaseSlide $presentation 2
[void](Add-TextBox $slide 70 70 780 70 'Why this problem matters' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 900 40 'Wind energy and biodiversity share the same airspace.' 20 $false $MUTED)
Add-CardTitle $slide 70 210 350 'Ecology' 'Bird strikes are a real conservation issue, especially during migration peaks and for high-impact groups.' $ORANGE
Add-CardTitle $slide 445 210 350 'Operations' 'Turbine shutdowns are costly, so mitigation needs to be targeted, trustworthy, and defensible.' $TEAL
Add-CardTitle $slide 820 210 350 'Monitoring' 'Radar gives scale and range, but bird-group classification is hard because data are noisy, imbalanced, and seasonally shifting.' $GOLD
[void](Add-TextBox $slide 70 460 1040 120 'Our opportunity was not just to classify better.
It was to turn difficult sensor data into practical decision support for nature-inclusive wind energy.' 24 $true $TEXT 'Segoe UI Semibold')

# Slide 3
$slide = Add-BaseSlide $presentation 3
[void](Add-TextBox $slide 70 70 900 70 'What made our team different' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 980 40 'We built a disciplined research workflow, not a one-off leaderboard model.' 20 $false $MUTED)
$steps = @(
    @{T='Hypothesis'; B='Start from a technical or ecological idea'},
    @{T='Experiment'; B='Run controlled tests and compare variants'},
    @{T='Honest validation'; B='Check robustness under month / seasonal shift'},
    @{T='Keep or reject'; B='Preserve only what generalizes'},
    @{T='Deployable system'; B='Turn insights into a defendable pipeline'}
)
$left = 60
for ($i=0; $i -lt $steps.Count; $i++) {
    $box = Add-FillRect $slide ($left + $i*235) 250 210 170 $BG2 $BG2 16
    $box.Line.ForeColor.RGB = if ($i % 2 -eq 0) { $ORANGE } else { $TEAL }
    $box.Line.Weight = 1.4
    [void](Add-TextBox $slide ($left + 15 + $i*235) 270 180 30 ($steps[$i].T) 20 $true $TEXT 'Segoe UI Semibold' 2)
    [void](Add-TextBox $slide ($left + 15 + $i*235) 315 180 70 ($steps[$i].B) 14 $false $MUTED 'Segoe UI' 2)
    if ($i -lt $steps.Count-1) {
        $arrow = $slide.Shapes.AddShape(33, ($left + 198 + $i*235), 315, 35, 35)
        $arrow.Fill.ForeColor.RGB = $ORANGE
        $arrow.Line.Visible = 0
    }
}
[void](Add-TextBox $slide 80 500 1120 120 'This workflow gave us two advantages: speed and judgment.
We iterated aggressively, tracked what failed, and even won the competition award for the most submissions — not as brute force, but as disciplined learning.' 22 $true $TEXT 'Segoe UI Semibold')

# Slide 4
$slide = Add-BaseSlide $presentation 4
[void](Add-TextBox $slide 70 70 900 70 'System architecture' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 980 40 'Simple idea: combine movement, radar, and environmental context into one robust decision pipeline.' 19 $false $MUTED)
$boxes = @(
    @{L=70; T='Inputs'; B='Radar trajectories
+ external context'; C=$ORANGE},
    @{L=320; T='Feature stack'; B='Kinematics
Log-signatures
catch22
validated context'; C=$TEAL},
    @{L=570; T='Models'; B='Ranking-aware ensemble
+ multiclass probability model'; C=$GOLD},
    @{L=820; T='Output'; B='Probabilistic bird-group ranking
for operator / ecologist review'; C=$RED}
)
foreach ($b in $boxes) {
    $r = Add-FillRect $slide $b.L 260 185 180 $BG2 $BG2 18
    $r.Line.ForeColor.RGB = $b.C
    $r.Line.Weight = 1.6
    [void](Add-TextBox $slide ($b.L+15) 282 155 28 $b.T 20 $true $TEXT 'Segoe UI Semibold' 2)
    [void](Add-TextBox $slide ($b.L+15) 328 155 85 $b.B 15 $false $MUTED 'Segoe UI' 2)
}
for ($i=0; $i -lt 3; $i++) {
    $arrow = $slide.Shapes.AddShape(33, 265 + $i*250, 332, 38, 38)
    $arrow.Fill.ForeColor.RGB = $ORANGE
    $arrow.Line.Visible = 0
}
$callout = Add-FillRect $slide 1035 245 180 210 $BG2 $BG2 16
$callout.Line.ForeColor.RGB = $TEAL
$callout.Line.Weight = 1.6
[void](Add-TextBox $slide 1052 265 145 28 'Critical constraint' 18 $true $TEXT 'Segoe UI Semibold' 2)
[void](Add-TextBox $slide 1052 308 145 120 'The hard part was not only class imbalance.
It was month / seasonal shift.
So we designed for generalization, not just local score gains.' 15 $false $MUTED 'Segoe UI' 2)

# Slide 5
$slide = Add-BaseSlide $presentation 5
[void](Add-TextBox $slide 70 70 900 70 'What actually moved the needle' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 980 40 'Our best gains came from changing how we thought about the problem.' 19 $false $MUTED)
Add-CardTitle $slide 70 220 350 '1. Ranking, not just classification' 'Macro-mAP rewards ordering. Once we treated this as a ranking problem, our modeling choices became much better aligned with the metric.' $ORANGE
Add-CardTitle $slide 445 220 350 '2. Shift was a first-class failure mode' 'Some ideas looked good locally but collapsed across months. Honest validation changed our feature selection and model strategy.' $TEAL
Add-CardTitle $slide 820 220 350 '3. Judgment beat tricks' 'We rejected attractive but brittle methods and kept the pieces we could explain, defend, and realistically deploy.' $GOLD
[void](Add-TextBox $slide 80 500 1110 90 'That is the strongest message of our work: the workflow created the architecture.' 24 $true $TEXT 'Segoe UI Semibold')

# Slide 6
$slide = Add-BaseSlide $presentation 6
[void](Add-TextBox $slide 70 70 900 70 'Why we trust this result' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 980 40 'We wanted a system we could explain to a jury, to domain experts, and to future operators.' 19 $false $MUTED)
Add-CardTitle $slide 70 220 250 'Finalist-level system' 'Competitive enough to reach the AI Cup 2026 finals.' $ORANGE
Add-CardTitle $slide 345 220 250 '200+ tracked iterations' 'Our repo reflects broad exploration rather than one lucky run.' $TEAL
Add-CardTitle $slide 620 220 250 'Most submissions award' 'A signal of disciplined throughput and rapid learning.' $GOLD
Add-CardTitle $slide 895 220 250 'Honest evaluation' 'Cross-month thinking reduced overclaiming and improved credibility.' $RED
[void](Add-TextBox $slide 80 500 1110 100 'The point is not that we solved everything.
The point is that we built a stronger, more defendable system by combining experimentation discipline with deployment thinking.' 22 $true $TEXT 'Segoe UI Semibold')

# Slide 7
$slide = Add-BaseSlide $presentation 7
[void](Add-TextBox $slide 70 70 900 70 'How this gets used in the real world' 28 $true $TEXT 'Segoe UI Semibold')
[void](Add-TextBox $slide 70 130 980 40 'The model is most valuable as decision support inside a monitoring and mitigation workflow.' 19 $false $MUTED)
$flow = @(
    @{L=70; T='Radar monitoring'; C=$ORANGE},
    @{L=320; T='AI ranking + uncertainty'; C=$TEAL},
    @{L=570; T='Operator / ecologist review'; C=$GOLD},
    @{L=820; T='Targeted mitigation action'; C=$RED}
)
foreach ($f in $flow) {
    $r = Add-FillRect $slide $f.L 255 200 95 $BG2 $BG2 18
    $r.Line.ForeColor.RGB = $f.C
    $r.Line.Weight = 1.6
    [void](Add-TextBox $slide ($f.L+15) 288 170 50 $f.T 17 $true $TEXT 'Segoe UI Semibold' 2)
}
for ($i=0; $i -lt 3; $i++) {
    $arrow = $slide.Shapes.AddShape(33, 270 + $i*250, 296, 38, 38)
    $arrow.Fill.ForeColor.RGB = $ORANGE
    $arrow.Line.Visible = 0
}
[void](Add-TextBox $slide 70 430 1140 130 'We believe the strength of our work is not only that it performs well,
but that it is rigorous, ecologically grounded, and realistically deployable.

That is what we hope you remember about our team.' 26 $true $TEXT 'Segoe UI Semibold' 2)

$outputFull = Join-Path (Get-Location) $OutputPath
$presentation.SaveAs($outputFull)
$presentation.Close()
$ppt.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($presentation) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
[GC]::Collect(); [GC]::WaitForPendingFinalizers()
Write-Output "Saved: $outputFull"
