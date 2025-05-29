var tokens = [];
var index = 0;
var outlet = outlet || function() {};

function loadbang() {
    var f = new File("generatedTokens.json", "read");
    var jsonString = "";

    while (f.position < f.eof) {
        jsonString += f.readstring(f.eof - f.position);
    }
    f.close();

    tokens = JSON.parse(jsonString);
    post("Loaded", tokens.length, "tokens\n");
    index = 0;
    next();
}

function next() {
    if (index >= tokens.length) {
        post("Sequence done\n");
        return;
    }

    var token = tokens[index];
    index++;

    if (token.startsWith("time_shift_")) {
        var delay = parseInt(token.split("_")[2]);
        // Schedule next token after delay
        task = new Task(next, this);
        task.schedule(delay);
    } else if (token.startsWith("note_on_")) {
        var pitch = parseInt(token.split("_")[2]);
        outlet(0, ["note_on", pitch, 100]);
        next(); // Immediately continue
    } else if (token.startsWith("note_off_")) {
        var pitch = parseInt(token.split("_")[2]);
        outlet(0, ["note_off", pitch, 0]);
        next(); // Immediately continue
    } else {
        next(); // Skip unknown tokens
    }
}
