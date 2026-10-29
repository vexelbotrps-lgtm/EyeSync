$ErrorActionPreference = "Stop"
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

function Write-VoiceEvent($event) {
    $json = ConvertTo-Json -InputObject $event -Compress -Depth 4
    [Console]::Out.WriteLine($json)
    [Console]::Out.Flush()
}

try {
    Add-Type -AssemblyName System.Speech
    $recognizers = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
    $recognizer = $recognizers | Where-Object { $_.Culture.Name -eq "en-US" } | Select-Object -First 1
    if ($null -eq $recognizer) {
        throw "No installed en-US Windows speech recognizer was found."
    }

    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($recognizer)
    $engine.SetInputToDefaultAudioDevice()

    $phrases = @(
        "open the on screen keyboard",
        "close the on screen keyboard",
        "increase the size of the on screen keyboard",
        "decrease the size of the on screen keyboard",
        "let me adjust the position of the on screen keyboard",
        "done adjusting the position of the on screen keyboard"
    )
    $choices = New-Object System.Speech.Recognition.Choices
    foreach ($phrase in $phrases) {
        [void]$choices.Add($phrase)
    }
    $builder = New-Object System.Speech.Recognition.GrammarBuilder
    $builder.Culture = $recognizer.Culture
    $builder.Append($choices)
    $commandGrammar = New-Object System.Speech.Recognition.Grammar($builder)
    $commandGrammar.Name = "Commands"
    $engine.LoadGrammar($commandGrammar)

    $dictationGrammar = New-Object System.Speech.Recognition.DictationGrammar
    $dictationGrammar.Name = "Dictation"
    $dictationGrammar.Enabled = $false
    $engine.LoadGrammar($dictationGrammar)

    $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $handler = {
        $result = $Event.SourceEventArgs.Result
        $grammarName = $result.Grammar.Name
        $kind = if ($grammarName -eq "Commands") { "command" } else { "dictation" }
        $event = @{
            kind = $kind
            text = $result.Text
            confidence = [Math]::Round($result.Confidence, 3)
        }
        $json = ConvertTo-Json -InputObject $event -Compress -Depth 4
        [Console]::Out.WriteLine($json)
        [Console]::Out.Flush()
    }
    $activityHandler = {
        [Console]::Out.WriteLine('{"kind":"speech_detected"}')
        [Console]::Out.Flush()
    }
    $rejectedHandler = {
        $confidence = [Math]::Round(
            $Event.SourceEventArgs.Result.Confidence, 3
        )
        $json = ConvertTo-Json -InputObject @{
            kind = "speech_rejected"
            confidence = $confidence
        } -Compress
        [Console]::Out.WriteLine($json)
        [Console]::Out.Flush()
    }
    Register-ObjectEvent -InputObject $engine -EventName SpeechRecognized `
        -SourceIdentifier VoiceSpeechRecognized -Action $handler | Out-Null
    Register-ObjectEvent -InputObject $engine -EventName SpeechDetected `
        -SourceIdentifier VoiceSpeechDetected -Action $activityHandler | Out-Null
    Register-ObjectEvent -InputObject $engine -EventName SpeechRecognitionRejected `
        -SourceIdentifier VoiceSpeechRejected -Action $rejectedHandler | Out-Null
    $engine.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)
    Write-VoiceEvent @{
        kind = "ready"
        recognizer = $recognizer.Name
        culture = $recognizer.Culture.Name
    }

    $inputReader = New-Object System.IO.StreamReader(
        [Console]::OpenStandardInput(), [System.Text.Encoding]::UTF8
    )
    $inputTask = $inputReader.ReadLineAsync()
    $shutdown = $false
    while (-not $shutdown) {
        Wait-Event -Timeout 0.1 | Out-Null
        if (-not $inputTask.IsCompleted) {
            continue
        }
        $line = $inputTask.GetAwaiter().GetResult()
        if ($null -eq $line) {
            break
        }
        try {
            $control = ConvertFrom-Json -InputObject $line
            if ($control.action -eq "dictation") {
                $dictationGrammar.Enabled = [bool]$control.enabled
            }
            elseif ($control.action -eq "speak") {
                if (-not [string]::IsNullOrWhiteSpace($control.text)) {
                    [void]$synthesizer.SpeakAsync([string]$control.text)
                }
            }
            elseif ($control.action -eq "shutdown") {
                $shutdown = $true
            }
        }
        catch {
            Write-VoiceEvent @{ kind = "error"; message = $_.Exception.Message }
        }
        if (-not $shutdown) {
            $inputTask = $inputReader.ReadLineAsync()
        }
    }
}
catch {
    Write-VoiceEvent @{ kind = "error"; message = $_.Exception.Message }
    exit 1
}
finally {
    Unregister-Event -SourceIdentifier VoiceSpeechRecognized -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier VoiceSpeechDetected -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier VoiceSpeechRejected -ErrorAction SilentlyContinue
    if ($null -ne $engine) {
        $engine.RecognizeAsyncCancel()
        $engine.Dispose()
    }
    if ($null -ne $synthesizer) {
        $synthesizer.Dispose()
    }
    if ($null -ne $inputReader) {
        $inputReader.Dispose()
    }
}
