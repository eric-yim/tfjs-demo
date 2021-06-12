## If you are developing posenet locally, and want to test the changes in the demos

Cd into the posenet folder:
```sh
cd posenet
```

Install dependencies:
```sh
yarn
```

Publish posenet locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
```

Link the local posenet to the demo:
```sh
yarn yalc link @tensorflow-models/posenet
```

Start the dev demo server:
```sh
yarn start
```
